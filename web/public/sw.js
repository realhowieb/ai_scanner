// HSF service worker: shows alert notifications pushed by the API (api/webpush.py) and
// opens the app on a click. It caches nothing and never touches page requests.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

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
