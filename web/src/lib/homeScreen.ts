// Home Screen app helpers: is HSF running as the installed app, and could it be installed here.

/** True inside the HSF iPhone app (mobile/, Capacitor), not a browser. */
export function isNativeApp(): boolean {
  if (typeof window === "undefined") return false;
  const cap = (window as unknown as { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor;
  return cap?.isNativePlatform?.() === true;
}

/** True when opened from the Home Screen (no Safari bars) or as the iPhone app, rather than in a browser tab. */
export function isStandalone(): boolean {
  if (typeof window === "undefined") return false;
  if (isNativeApp()) return true;
  const nav = navigator as Navigator & { standalone?: boolean };
  return nav.standalone === true || window.matchMedia?.("(display-mode: standalone)").matches === true;
}

/** True on an iPhone, where Add to Home Screen is the only way to get HSF's alerts. */
export function isIPhone(): boolean {
  return typeof navigator !== "undefined" && /iPhone|iPod/.test(navigator.userAgent);
}

const HINT_KEY = "hsf-home-screen-hint-dismissed";

export function hintDismissed(): boolean {
  try {
    return localStorage.getItem(HINT_KEY) === "1";
  } catch {
    return false;
  }
}

export function dismissHint(): void {
  try {
    localStorage.setItem(HINT_KEY, "1");
  } catch {
    // private mode: the hint just comes back next visit
  }
}
