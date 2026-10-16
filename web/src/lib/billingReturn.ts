// Local reminder only. Entitlements always come from the authenticated API.
const KEY = "hsf_billing_pending";
export const BILLING_STARTED = "hsf-billing-started";
type Pending = { at: number; previous: string | null; expected: string | null };

export function pendingBilling(): Pending | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(KEY) ?? "null") as Pending | null;
    return value && Number.isFinite(value.at) && Date.now() - value.at < 2 * 60 * 60_000 ? value : null;
  } catch { return null; }
}

export function beginBilling(previous: string | null, expected: string | null = null): void {
  try { sessionStorage.setItem(KEY, JSON.stringify({ at: Date.now(), previous, expected })); } catch { /* optional reminder */ }
  window.dispatchEvent(new Event(BILLING_STARTED));
}

export function finishBilling(): void {
  try { sessionStorage.removeItem(KEY); } catch { /* optional reminder */ }
}
