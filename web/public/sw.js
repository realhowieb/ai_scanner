// HSF service worker: shows alert notifications pushed by the API (api/webpush.py) and
// opens the app on a click. The only thing it caches is the offline screen: a page load
// that can't reach the server shows /offline.html. Every other request goes straight
// to the network untouched.
const OFFLINE_CACHE = "hsf-offline-v1";
const OFFLINE_URL = "/offline.html";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(OFFLINE_CACHE).then((c) => c.add(new Request(OFFLINE_URL, { cache: "reload" }))).catch(() => undefined));
  self.skipWaiting();
});
self.addEventListener("activate", (event) => event.waitUntil((async () => {
  for (const key of await caches.keys()) if (key.startsWith("hsf-offline-") && key !== OFFLINE_CACHE) await caches.delete(key);
  await self.clients.claim();
})()));

self.addEventListener("fetch", (event) => {
  if (event.request.mode !== "navigate") return;
  event.respondWith(fetch(event.request).catch(async () => (await caches.match(OFFLINE_URL)) || Response.error()));
});

self.addEventListener("push", (event) => {
  let msg = {};
  try {
    msg = event.data ? event.data.json() : {};
  } catch {
    msg = { body: event.data ? event.data.text() : "" };
  }
  const url = typeof msg.url === "string" && msg.url.startsWith("/") ? msg.url : "/alerts";
  event.waitUntil(self.registration.showNotification(msg.title || "HSF alert", {
    body: msg.body || "One of your alerts fired.",
    tag: msg.tag || "hsf-alert",
    renotify: true,
    icon: "/icon.svg",
    data: { url },
  }));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = new URL((event.notification.data && event.notification.data.url) || "/alerts", self.location.origin).href;
  event.waitUntil((async () => {
    const open = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const c of open) {
      if (new URL(c.url).origin === self.location.origin && "focus" in c) {
        await c.focus();
        if ("navigate" in c) await c.navigate(url).catch(() => undefined);
        return;
      }
    }
    await self.clients.openWindow(url);
  })());
});
