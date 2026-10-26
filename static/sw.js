const CACHE_NAME = 'waste-ai-cache-v2';
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/icons/icon-96.png',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  
  // Never intercept or cache MJPEG streams, snapshots, or dynamic polling APIs
  if (
    url.pathname.startsWith('/video_feed') ||
    url.pathname.startsWith('/camera_snapshot') ||
    url.pathname.startsWith('/api/') ||
    url.pathname === '/status'
  ) {
    return;
  }

  // Network first, fall back to cache
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
