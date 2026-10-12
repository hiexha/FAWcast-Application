"""
train_fawcast.py - retrain the FAWcast models and write model.json for the app
=============================================================================

What it does
  1. Reads the BPI Fall Armyworm report file (Excel), the list of modeled
     provinces (provinces.csv) and monthly NASA POWER weather (weather_monthly.csv).
  2. Builds the province-quarter dataset exactly as in the paper:
       outbreak (Y)  = BPI reports from 2 or more municipalities in the quarter
       lagY          = outbreak in the same province in the previous quarter
       nbR           = outbreak in another province of the same region, previous quarter
       larea1        = ln(1 + BPI "Area Planted" in the province, previous quarter)
       lnat1         = ln(1 + number of BPI reports nationwide, previous quarter)
       tmin/rain/hum = averages of the previous quarter (NASA POWER)
       Q2, Q3, Q4    = season of the quarter being predicted (Q1 is the reference)
  3. Fits the hierarchical logistic regression (region and province random
     intercepts, Penalized Quasi-Likelihood) for
       - the quarterly forecast, and
       - the monthly update (adds first-month report counts; outcome = outbreak
         in months 2-3 of the quarter).
  4. Validates both models (5-fold CV, leave-one-province-out, leave-one-year-out,
     forward chaining) unless you add --quick.
  5. Writes model.json. Copy it next to index.html and the app uses it.

How to run (Python 3.9+)
  pip install numpy pandas scikit-learn openpyxl
  python train_fawcast.py --bpi FAW_2020-2025.xlsx

Options
  --bpi FILE          BPI report file (Excel). Required.
  --provinces FILE    default provinces.csv
  --weather FILE      default weather_monthly.csv (columns: Province, Year, Month, tmin, rain, hum)
  --out FILE          default model.json
  --quick             skip the slower validation (keeps the performance numbers from the old model.json)

When new data arrives
  - Append the new BPI rows to the Excel file (same columns).
  - Add the new months to weather_monthly.csv for every province in provinces.csv
    (NASA POWER, community AG, monthly: T2M_MIN in deg C, PRECTOTCORR in mm/day, RH2M in %).
  - Run the script again and replace model.json in the app folder.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

MONTHS = {m: i for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                       'August', 'September', 'October', 'November', 'December'], 1)}
F = ['tmin_l1', 'rain_l1', 'hum_l1', 'Q2', 'Q3', 'Q4', 'lagY', 'nbR', 'larea1', 'lnat1']
FM = F + ['m1p', 'm1r', 'm1n']
REGION_NAMES = {'CAR': 'Cordillera Administrative Region', 'I': 'Region I (Ilocos)', 'II': 'Region II (Cagayan Valley)',
                'III': 'Region III (Central Luzon)', 'IV-A': 'Region IV-A (CALABARZON)', 'IV-B': 'Region IV-B (MIMAROPA)',
                'V': 'Region V (Bicol)', 'VI': 'Region VI (Western Visayas)', 'NIR': 'Negros Island Region',
                'VII': 'Region VII (Central Visayas)', 'VIII': 'Region VIII (Eastern Visayas)', 'IX': 'Region IX (Zamboanga Peninsula)',
                'X': 'Region X (Northern Mindanao)', 'XI': 'Region XI (Davao)', 'XII': 'Region XII (SOCCSKSARGEN)',
                'XIII': 'Region XIII (Caraga)', 'BARMM': 'BARMM'}
TIERS = [{'max': 0.30, 'name': 'Low', 'action': 'Continue routine field monitoring and standard pest management. Teach farmers to recognise and report Fall Armyworm.'},
         {'max': 0.50, 'name': 'Moderate', 'action': 'Increase field inspections, set up pheromone traps for early detection, and prepare control materials with the local agriculture office.'},
         {'max': 0.70, 'name': 'High', 'action': 'Start close monitoring in affected areas and apply integrated pest management promptly once infestation is confirmed.'},
         {'max': 1.01, 'name': 'Very High', 'action': 'Issue an outbreak advisory, intensify surveillance in nearby farms, and coordinate rapid intervention with agricultural authorities.'}]


def nice(p):
    return p.replace('Mt. Province', 'Mountain Province').replace(' Del ', ' del ').replace(' De ', ' de ')


def qlabel(t):
    return f"{2020 + (t - 1) // 4} Q{(t - 1) % 4 + 1}"


# ---------------------------------------------------------------- model fitting
def fit2(X, y, rc, pc, nr, npv):
    """Hierarchical logistic regression with region and province random intercepts (PQL)."""
    n = len(y); Xd = np.column_stack([np.ones(n), X]); pf = Xd.shape[1]
    Zr = np.zeros((n, nr)); Zp = np.zeros((n, npv))
    Zr[np.arange(n), rc] = 1; Zp[np.arange(n), pc] = 1
    M = np.column_stack([Xd, Zr, Zp]); nt = M.shape[1]; th = np.zeros(nt); sr = sp = 1.0
    for _ in range(150):
        eta = np.clip(M @ th, -30, 30); p = 1 / (1 + np.exp(-eta)); wt = np.clip(p * (1 - p), 1e-6, None)
        yw = eta + (y - p) / wt
        dd = np.zeros(nt); dd[pf:pf + nr] = 1 / sr; dd[pf + nr:] = 1 / sp
        A = (M.T * wt) @ M + np.diag(dd); b = (M.T * wt) @ yw; tn = np.linalg.solve(A, b)
        diff = np.abs(tn - th).max(); th = tn; cov = np.linalg.inv(A)
        ur = th[pf:pf + nr]; up = th[pf + nr:]
        srn = max((ur @ ur + np.trace(cov[pf:pf + nr, pf:pf + nr])) / nr, 1e-3)
        spn = max((up @ up + np.trace(cov[pf + nr:, pf + nr:])) / npv, 1e-3)
        if diff < 1e-7 and abs(srn - sr) < 1e-7 and abs(spn - sp) < 1e-7:
            sr, sp = srn, spn; break
        sr, sp = srn, spn
    return th[:pf], th[pf:pf + nr], th[pf + nr:], sr, sp


def lin(b, ur, up, Z, r, p, use_r=True, use_p=True):
    return np.column_stack([np.ones(len(Z)), Z]) @ b + (ur[r] if use_r else 0) + (up[p] if use_p else 0)


# ---------------------------------------------------------------- data
def build(bpi_path, prov_path, weather_path):
    d = pd.read_excel(bpi_path)
    d = d.loc[:, ~d.columns.astype(str).str.startswith('Unnamed')]
    d['Region'] = d['Region'].astype(str).str.strip().str.upper()
    d['Province'] = d['Province'].astype(str).str.replace('\xa0', ' ').str.strip().str.title()
    d['M'] = d['Month'].astype(str).str.strip().str.title().map(MONTHS)
    d = d.dropna(subset=['M', 'Year']).copy()
    d['Year'] = d['Year'].astype(int); d['M'] = d['M'].astype(int)
    d['Q'] = (d.M - 1) // 3 + 1; d['t'] = (d.Year - 2020) * 4 + d.Q; d['mq'] = (d.M - 1) % 3 + 1
    d['area'] = pd.to_numeric(d['Area Planted'], errors='coerce')

    prov = pd.read_csv(prov_path)
    prov['Province'] = prov.Province.str.strip()
    w = pd.read_csv(weather_path); w['Province'] = w.Province.astype(str).str.replace('\xa0', ' ').str.strip()
    missing = set(prov.Province) - set(w.Province)
    if missing:
        sys.exit(f"weather_monthly.csv has no rows for: {sorted(missing)}")
    w['Q'] = (w.Month - 1) // 3 + 1; w['t'] = (w.Year - 2020) * 4 + w.Q
    wq = w.groupby(['Province', 't'])[['tmin', 'rain', 'hum']].mean()

    g = d.groupby(['Region', 'Province', 't']).Municipality.nunique().reset_index(name='muni')
    g['out'] = (g.muni >= 2).astype(int)
    # outbreaks are counted within the province's own region label, as in the paper
    outbreak = set(zip(g.Region[g.out == 1], g.Province[g.out == 1], g.t[g.out == 1]))
    byreg = {}
    for r, p, t in zip(g.Region[g.out == 1], g.Province[g.out == 1], g.t[g.out == 1]):
        byreg.setdefault((r, t), set()).add(p)
    MUNI = {(r, p, t): m for r, p, t, m in zip(g.Region, g.Province, g.t, g.muni)}
    AREA = d.groupby(['Province', 't']).area.sum().to_dict()
    NAT = d.groupby('t').size().to_dict()
    M1 = d[d.mq == 1].groupby(['Province', 't']).size().to_dict()
    M1R = d[d.mq == 1].groupby(['Region', 't']).size().to_dict()
    M1N = d[d.mq == 1].groupby('t').size().to_dict()
    M23 = d[d.mq > 1].groupby(['Province', 't']).Municipality.nunique().to_dict()
    T_last = int(d.t.max())

    rows = []
    for p, r in zip(prov.Province, prov.Region):
        for t in range(2, T_last + 1):
            if (p, t - 1) not in wq.index:
                continue
            wl = wq.loc[(p, t - 1)]; qq = (t - 1) % 4 + 1
            rows.append(dict(Region=r, Province=p, t=t, Year=2020 + (t - 1) // 4, Q=qq,
                             tmin_l1=wl.tmin, rain_l1=wl.rain, hum_l1=wl.hum,
                             Q2=int(qq == 2), Q3=int(qq == 3), Q4=int(qq == 4),
                             Y=int((r, p, t) in outbreak), muni=int(MUNI.get((r, p, t), 0)),
                             nbR=float(len(byreg.get((r, t - 1), set()) - {p}) > 0),
                             larea1=np.log1p(AREA.get((p, t - 1), 0) or 0), lnat1=np.log1p(NAT.get(t - 1, 0)),
                             m1p=np.log1p(M1.get((p, t), 0)), m1r=np.log1p(M1R.get((r, t), 0)), m1n=np.log1p(M1N.get(t, 0)),
                             y23=int(M23.get((p, t), 0) >= 2)))
    q = pd.DataFrame(rows).sort_values(['Province', 't']).reset_index(drop=True)
    q['lagY'] = q.groupby('Province').Y.shift(1)
    q.loc[q.groupby('Province').t.diff() != 1, 'lagY'] = np.nan
    q = q.dropna(subset=['lagY']).reset_index(drop=True)

    # latest quarter: inputs for forecasting the next quarter
    latest = {}
    for p, r in zip(prov.Province, prov.Region):
        if (p, T_last) not in wq.index:
            continue
        wl = wq.loc[(p, T_last)]
        latest[nice(p)] = dict(tmin=round(float(wl.tmin), 2), rain=round(float(wl.rain), 2), hum=round(float(wl.hum), 2),
                               lagY=int((r, p, T_last) in outbreak), nbR=int(len(byreg.get((r, T_last), set()) - {p}) > 0),
                               area=round(float(AREA.get((p, T_last), 0) or 0), 1))
    return q, latest, NAT, T_last


# ---------------------------------------------------------------- validation
def validate(q, feats, yname, nr, npv, rc, pc):
    y = q[yname].values; X = q[feats].values.astype(float); Z = (X - X.mean(0)) / X.std(0); yrs = q.Year.values
    ps = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Z, y):
        b, a, c, _, _ = fit2(Z[tr], y[tr], rc[tr], pc[tr], nr, npv); ps[te] = lin(b, a, c, Z[te], rc[te], pc[te])
    pl = np.zeros(len(y))
    for pv in range(npv):
        te = pc == pv; b, a, c, _, _ = fit2(Z[~te], y[~te], rc[~te], pc[~te], nr, npv); pl[te] = lin(b, a, c, Z[te], rc[te], pc[te], True, False)
    py = np.zeros(len(y))
    for Y in sorted(set(yrs)):
        te = yrs == Y; b, a, c, _, _ = fit2(Z[~te], y[~te], rc[~te], pc[~te], nr, npv); py[te] = lin(b, a, c, Z[te], rc[te], pc[te])
    fw, fy = [], []
    for Y in sorted(set(yrs))[2:]:
        tr = yrs < Y; te = yrs == Y; b, a, c, _, _ = fit2(Z[tr], y[tr], rc[tr], pc[tr], nr, npv)
        fw += list(lin(b, a, c, Z[te], rc[te], pc[te])); fy += list(y[te])
    return {'cv_auc': round(roc_auc_score(y, ps), 3), 'new_province_auc': round(roc_auc_score(y, pl), 3),
            'new_year_auc': round(roc_auc_score(y, py), 3), 'forward_auc': round(roc_auc_score(fy, fw), 3),
            'cv_accuracy': round(accuracy_score(y, ps > 0), 3)}


def fit_export(q, feats, yname, regs, provs, rc, pc):
    y = q[yname].values; X = q[feats].values.astype(float); mu = X.mean(0); sd = X.std(0); Z = (X - mu) / sd
    b, ur, up, _, _ = fit2(Z, y, rc, pc, len(regs), len(provs))
    coef = {'intercept': float(b[0] - np.sum(b[1:] * mu / sd))}
    coef.update({f: float(v) for f, v in zip(feats, b[1:] / sd)})
    return {'coef': coef, 'u_region': {r: float(v) for r, v in zip(regs, ur)},
            'u_province': {nice(p): float(v) for p, v in zip(provs, up)}}


def main():
    ap = argparse.ArgumentParser(description='Retrain FAWcast and write model.json')
    ap.add_argument('--bpi', required=True); ap.add_argument('--provinces', default='provinces.csv')
    ap.add_argument('--weather', default='weather_monthly.csv'); ap.add_argument('--out', default='model.json')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    q, latest, NAT, T_last = build(a.bpi, a.provinces, a.weather)
    regs = sorted(q.Region.unique()); provs = sorted(q.Province.unique())
    rc = q.Region.map({r: i for i, r in enumerate(regs)}).values; pc = q.Province.map({p: i for i, p in enumerate(provs)}).values
    print(f"Dataset: {len(q):,} province-quarters, {len(provs)} provinces, {len(regs)} regions, "
          f"{qlabel(int(q.t.min()))} to {qlabel(T_last)}, outbreak share {q.Y.mean():.1%}")

    model = {'name': 'FAWcast', 'version': pd.Timestamp.today().strftime('%Y.%m'),
             'trained_on': f"BPI Fall Armyworm reports and NASA POWER weather, {len(provs)} provinces in {len(regs)} regions, "
                           f"{qlabel(int(q.t.min()))} to {qlabel(T_last)} ({len(q):,} province-quarters)"}
    model['quarterly'] = fit_export(q, F, 'Y', regs, provs, rc, pc)
    model['monthly'] = fit_export(q, FM, 'y23', regs, provs, rc, pc)

    old = {}
    if os.path.exists(a.out):
        try:
            old = json.load(open(a.out))
        except Exception:
            old = {}
    if a.quick:
        model['quarterly']['performance'] = old.get('quarterly', {}).get('performance', {})
        model['monthly']['performance'] = old.get('monthly', {}).get('performance', {})
        print('Validation skipped (--quick).')
    else:
        print('Validating the quarterly model (this takes a few minutes)...')
        model['quarterly']['performance'] = validate(q, F, 'Y', len(regs), len(provs), rc, pc)
        print('  ', model['quarterly']['performance'])
        print('Validating the monthly update...')
        model['monthly']['performance'] = validate(q, FM, 'y23', len(regs), len(provs), rc, pc)
        print('  ', model['monthly']['performance'])

    model['province_region'] = {nice(p): q.loc[q.Province == p, 'Region'].iloc[0] for p in provs}
    ex = lambda c: float(np.expm1(c))
    model['ranges'] = {'tmin': [float(q.tmin_l1.min()), float(q.tmin_l1.max())], 'rain': [float(q.rain_l1.min()), float(q.rain_l1.max())],
                       'hum': [float(q.hum_l1.min()), float(q.hum_l1.max())], 'area': [0, ex(q.larea1.max())],
                       'national': [ex(q.lnat1.min()), ex(q.lnat1.max())], 'm1p': [0, ex(q.m1p.max())],
                       'm1r': [0, ex(q.m1r.max())], 'm1n': [ex(q.m1n.min()), ex(q.m1n.max())]}
    model['tiers'] = TIERS
    model['latest'] = {'based_on': qlabel(T_last), 'forecast_for': qlabel(T_last + 1),
                       'national_reports': int(NAT.get(T_last, 0)), 'provinces': latest}
    model['national_reports_by_quarter'] = {qlabel(t): int(v) for t, v in sorted(NAT.items())}
    model['means'] = {k: float(q[k].mean()) for k in FM}
    model['region_names'] = REGION_NAMES
    # quarter-by-quarter history for the app's History page (model estimates use the final fitted model)
    cq = model['quarterly']; zc = cq['coef']['intercept'] + sum(cq['coef'][f] * q[f].values.astype(float) for f in F)
    zc = zc + q.Region.map(cq['u_region']).fillna(0).values + q.Province.map(lambda p: cq['u_province'].get(nice(p), 0)).values
    q['_p'] = 1 / (1 + np.exp(-zc))
    model['history'] = {'first': qlabel(int(q.t.min())), 'provinces': {
        nice(p): {'t': [int(v) for v in g.t], 'y': ''.join(str(int(v)) for v in g.Y), 'mu': [int(v) for v in g.muni],
                  'p': [round(float(v) * 100, 1) for v in g._p]} for p, g in q.sort_values('t').groupby('Province')}}
    json.dump(model, open(a.out, 'w'), indent=1)
    print(f"Wrote {a.out}. Latest data: {qlabel(T_last)}; the app will forecast {qlabel(T_last + 1)}.")


if __name__ == '__main__':
    main()
