// FAWcast offline support: keeps the app working without internet after the first visit.
// The page and model.json are fetched fresh when online (so updates show up right away) and served from the cache when offline.
const CACHE = 'fawcast-v2';
const FILES = ['./', './index.html', './model.json', './manifest.webmanifest', './icons/icon-192.png', './icons/icon-512.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES))); self.skipWaiting(); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  const fresh = e.request.mode === 'navigate' || url.pathname.endsWith('/') || url.pathname.endsWith('index.html') || url.pathname.endsWith('model.json');
  if (fresh && url.origin === location.origin) {
    e.respondWith(fetch(e.request, {cache: 'no-store'}).then(r => { if (r.ok) { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); } return r; })
      .catch(() => caches.match(e.request).then(hit => hit || caches.match('./index.html'))));
    return;
  }
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(r => {
    if (r.ok && url.origin === location.origin) { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); }
    return r;
  }).catch(() => caches.match('./index.html'))));
});
