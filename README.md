# FAWcast app: setup guide

FAWcast estimates the chance of a Fall Armyworm outbreak in a Philippine corn
province. It runs the trained model from the research paper directly in the
browser, so it works offline once installed.

## What is in this folder

| File | What it is |
|---|---|
| `index.html` | The app. The trained model is built in, so it works even on its own. |
| `model.json` | The trained model (coefficients, province and region effects, latest data). The app loads this file if it is present, so retraining only means replacing it. |
| `manifest.webmanifest`, `sw.js`, `icons/` | Make the app installable on phones and laptops and let it work offline. |
| `train_fawcast.py` | Retrains the model from the BPI report file and NASA POWER weather and writes a new `model.json`. |
| `provinces.csv` | The 55 modeled provinces, their regions and coordinates. |
| `weather_monthly.csv` | Monthly NASA POWER weather for the 55 provinces, 2020 to 2025. |

## 1. Try it on your computer (2 minutes)

Double-click `index.html`. It opens in your browser and works right away.
(Installing and offline use need step 2.)

## 2. Put it online for free with GitHub Pages (about 15 minutes)

The app needs an `https://` address to be installable.

1. Make a free account at github.com.
2. Click **New repository**. Name it `fawcast`, set it to **Public**, and click **Create repository**.
3. Click **uploading an existing file**. Drag in everything from this folder,
   including the `icons` folder, then click **Commit changes**.
4. Go to **Settings → Pages**. Under **Branch**, choose `main` and `/ (root)`, then **Save**.
5. Wait about a minute. Your app is at `https://YOUR-USERNAME.github.io/fawcast/`.

## 3. Install it

- **Android (Chrome):** open the link, tap the **⋮** menu, then **Add to Home screen** (or **Install app**).
- **iPhone (Safari):** open the link, tap **Share**, then **Add to Home Screen**.
- **Laptop (Chrome or Edge):** open the link and click the install icon in the address bar,
  or the **Install app** button at the bottom of the page.

After the first visit, the app opens without internet.

## 4. Make an Android app file (APK)

1. Go to https://www.pwabuilder.com and paste your GitHub Pages link.
2. Click **Package for stores**, then **Android**, then **Generate**.
3. Download the zip. Inside is an `.apk` file you can install on Android phones
   (allow "Install from unknown sources" when asked). The same zip has the file
   Google Play needs if you publish later (Google charges a one-time developer fee).

## 5. Retrain the model when new data arrives

The paper recommends refitting every quarter.

1. Install Python 3.9 or newer, then run:
   `pip install numpy pandas scikit-learn openpyxl`
2. Add the new BPI rows to the Excel file (same columns as before).
3. Add the new months to `weather_monthly.csv` for every province
   (NASA POWER, community AG, monthly: `T2M_MIN` in °C, `PRECTOTCORR` in mm/day,
   `RH2M` in %; coordinates are in `provinces.csv`).
4. Run:
   `python train_fawcast.py --bpi YOUR_BPI_FILE.xlsx`
   This takes a few minutes because it also re-checks the model's accuracy.
   Add `--quick` to skip the accuracy check.
5. Upload the new `model.json` to your GitHub repository, replacing the old one.
   The app picks it up the next time it opens with internet.

To add a province, add a row to `provinces.csv` and its weather to `weather_monthly.csv`, then retrain.

## How to use the app

The app has six pages (bottom bar on phones, top bar on computers):

- **Forecast:** choose a province and quarter, then enter what happened in the previous quarter.
  **Use latest data** fills in the newest values. Switch to **Monthly update** after the first month
  of a quarter and enter that month's BPI report counts.
- **Map:** every province colored by next-quarter risk. Switch to **Past outbreaks** to see how often
  each province had an outbreak. Tap a province for its details.
- **Guide:** simple steps for farmers at each risk level, how to spot Fall Armyworm, how to check a
  field, ways to control it, and who to contact. Based on the DA's Fall Armyworm leaflet.
- **Rankings:** all 55 provinces grouped by risk level. Filter by region or search by name.
- **History:** each quarter since 2020 for one province: whether an outbreak happened, how many
  municipalities reported, and the model's estimate. Also shows the riskiest season.
- **About:** how the model works, its accuracy, and what each of the ten inputs does.

Risk tiers: Low (below 30%), Moderate (30–49%), High (50–69%), Very High (70% and above).

## Accuracy (from the paper)

| Test | Quarterly forecast | Monthly update |
|---|---|---|
| 5-fold cross-validation AUC | 0.824 | 0.844 |
| Provinces not used in training | 0.799 | 0.831 |
| Years not used in training | 0.785 | 0.807 |
| Forecasting later years | 0.773 | 0.798 |

An outbreak means BPI reports from two or more municipalities in a quarter, so quiet
reporting can hide real infestations. Use FAWcast together with field scouting.
