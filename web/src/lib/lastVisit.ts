/** "New since your last visit" is remembered in this browser, as in the Streamlit app:
 * the API hands back a "seen:baseline" marker of market scan ids, which holds no
 * account data. Storage can be unavailable (private mode, blocked site data);
 * then every visit reads as a first visit. */
export const MARKER_KEY = "hsf_last_market_run";

export function readMarker(): { seen?: number; baseline?: number } {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(MARKER_KEY);
  } catch {
    return {};
  }
  const [seen, baseline] = (raw ?? "").split(":").map((x) => (/^\d+$/.test(x) ? Number(x) : undefined));
  return { ...(seen ? { seen } : {}), ...(baseline ? { baseline } : {}) };
}

export function writeMarker(marker: string | null | undefined): void {
  if (!marker) return;
  try {
    window.localStorage.setItem(MARKER_KEY, marker);
  } catch {
    /* not remembered; the next visit reads as a first visit */
  }
}
