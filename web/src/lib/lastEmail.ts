// The email of the last successful sign-in on this device, so the sign-in form can fill it
// in. Only the email (never a password or token); "Not you?" and the signed-out page's
// "Forget my email" remove it. Storage can be blocked (private mode), so every call is safe.
import { useSyncExternalStore } from "react";

const KEY = "hsf-last-email";
const listeners = new Set<() => void>();
const changed = () => listeners.forEach((l) => l());

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** The remembered email ("" on the server, before hydration, or when none). */
export function useLastEmail(): string {
  return useSyncExternalStore(subscribe, readLastEmail, () => "");
}

export function readLastEmail(): string {
  try {
    const v = localStorage.getItem(KEY) || "";
    return v.includes("@") && v.length <= 254 ? v : "";
  } catch {
    return "";
  }
}

export function rememberEmail(email: string | null | undefined): void {
  const v = (email || "").trim().toLowerCase();
  if (!v.includes("@") || v.length > 254) return;
  try {
    localStorage.setItem(KEY, v);
  } catch {
    /* storage blocked */
  }
  changed();
}

export function forgetEmail(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* storage blocked */
  }
  changed();
}
