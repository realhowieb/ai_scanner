// Alert notifications in the HSF iPhone app (mobile/, Capacitor). The app's native bridge
// exposes the PushNotifications plugin on window.Capacitor; the device token goes to the
// API (POST /v1/me/devices, provider apns), which pushes through Apple (api/apns.py).
import { api, unwrap } from "@/api/client";
import { isNativeApp } from "@/lib/homeScreen";

import type { PushState } from "./webPush";

type Permission = "prompt" | "prompt-with-rationale" | "granted" | "denied";
type Listener = { remove: () => Promise<void> };
type PushPlugin = {
  checkPermissions(): Promise<{ receive: Permission }>;
  requestPermissions(): Promise<{ receive: Permission }>;
  register(): Promise<void>;
  unregister(): Promise<void>;
  addListener(event: "registration", cb: (t: { value: string }) => void): Promise<Listener>;
  addListener(event: "registrationError", cb: (e: { error: string }) => void): Promise<Listener>;
  addListener(event: "pushNotificationActionPerformed", cb: (a: { notification: { data?: Record<string, unknown> } }) => void): Promise<Listener>;
};

const DEVICE_KEY = "hsf-native-push-device";

function plugin(): PushPlugin | null {
  if (!isNativeApp()) return null;
  const cap = (window as unknown as { Capacitor?: { Plugins?: { PushNotifications?: PushPlugin } } }).Capacitor;
  return cap?.Plugins?.PushNotifications ?? null;
}

function storedDevice(): number | null {
  try {
    const id = Number(localStorage.getItem(DEVICE_KEY));
    return Number.isInteger(id) && id > 0 ? id : null;
  } catch {
    return null;
  }
}

function storeDevice(id: number | null): void {
  try {
    if (id) localStorage.setItem(DEVICE_KEY, String(id));
    else localStorage.removeItem(DEVICE_KEY);
  } catch {
    // private storage unavailable: the app re-registers next time notifications are turned on
  }
}

export function nativePushAvailable(): boolean {
  return plugin() !== null;
}

/** Asks iOS for a device token. Resolves with it, or rejects if Apple refuses. */
async function deviceToken(push: PushPlugin): Promise<string> {
  const subs: Listener[] = [];
  try {
    return await new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("The iPhone didn't return a notification token. Try again.")), 15000);
      void push.addListener("registration", (t) => { clearTimeout(timer); resolve(t.value); }).then((l) => subs.push(l));
      void push.addListener("registrationError", (e) => { clearTimeout(timer); reject(new Error(e.error || "Notifications couldn't be set up.")); })
        .then((l) => subs.push(l));
      void push.register().catch(reject);
    });
  } finally {
    await Promise.all(subs.map((l) => l.remove().catch(() => undefined)));
  }
}

async function registerWithApi(push: PushPlugin): Promise<void> {
  const token = await deviceToken(push);
  const device = await unwrap(api.POST("/v1/me/devices", {
    body: { push_token: token, platform: "ios", provider: "apns", device_name: "HSF iPhone app" },
  }));
  storeDevice(device.id);
}

export async function currentNativePushState(): Promise<PushState> {
  const push = plugin();
  if (!push) return "unsupported";
  const { receive } = await push.checkPermissions();
  if (receive === "denied") return "denied";
  return receive === "granted" && storedDevice() ? "on" : "off";
}

export async function enableNativePush(): Promise<PushState> {
  const push = plugin();
  if (!push) return "unsupported";
  const { receive } = await push.requestPermissions();
  if (receive !== "granted") return receive === "denied" ? "denied" : "off";
  await registerWithApi(push);
  return "on";
}

export async function disableNativePush(): Promise<void> {
  const push = plugin();
  const id = storedDevice();
  storeDevice(null);
  if (push) await push.unregister().catch(() => undefined);
  if (id) await unwrap(api.DELETE("/v1/me/devices/{device_id}", { params: { path: { device_id: id } } })).catch(() => undefined);
}

/** At app start: refresh the token the API has (iOS can change it) and open the page a
 * tapped notification points to. Never throws. */
export async function startNativePush(navigate: (url: string) => void): Promise<void> {
  const push = plugin();
  if (!push) return;
  await push.addListener("pushNotificationActionPerformed", ({ notification }) => {
    const url = notification.data?.url;
    navigate(typeof url === "string" && url.startsWith("/") ? url : "/alerts");
  }).catch(() => undefined);
  try {
    if (storedDevice() && (await push.checkPermissions()).receive === "granted") await registerWithApi(push);
  } catch {
    // signed out or offline: the next start tries again
  }
}
