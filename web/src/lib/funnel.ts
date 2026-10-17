// Where signed-out visitors came from (utm tags and an outside referrer), kept from the
// first page they open so the sign-up can be credited to it, plus the anonymous funnel
// events the classic app records (landing visit, call-to-action click, sign-up started).
// Best effort: storage or network failures never affect the page.
import { newRequestId } from "@/api/client";

const KEY = "hsf-attribution";
const KEEP_MS = 30 * 24 * 3600 * 1000;
const UTM = ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"] as const;

export type Attribution = Partial<Record<(typeof UTM)[number] | "referrer" | "ref", string>>;

const REF_RE = /^[a-z0-9]{4,32}$/;

/** An invite code (?ref=) on this page; it wins over an older first visit, so the friend who
 * sent the latest link gets the credit. */
function inviteCode(href: string): string | null {
  try {
    const v = (new URL(href).searchParams.get("ref") || "").trim().toLowerCase();
    return REF_RE.test(v) ? v : null;
  } catch {
    return null;
  }
}

/** Read utm tags and an outside referrer from this page; the first visit that has any is kept for 30 days. */
export function captureAttribution(href = typeof window === "undefined" ? "" : window.location.href,
  referrer = typeof document === "undefined" ? "" : document.referrer): Attribution {
  const kept = storedAttribution();
  const ref = inviteCode(href);
  if (kept) {
    if (!ref || kept.ref === ref) return kept;
    const merged = { ...kept, ref };
    try {
      const raw = JSON.parse(localStorage.getItem(KEY) || "null") as { at?: number } | null;
      localStorage.setItem(KEY, JSON.stringify({ at: raw?.at ?? Date.now(), attribution: merged }));
    } catch {
      /* private mode */
    }
    return merged;
  }
  const found: Attribution = ref ? { ref } : {};
  try {
    const url = new URL(href);
    for (const k of UTM) {
      const v = url.searchParams.get(k);
      if (v) found[k] = v.slice(0, 120);
    }
    if (referrer && new URL(referrer).host !== url.host) found.referrer = referrer.slice(0, 200);
  } catch {
    /* no attribution */
  }
  if (Object.keys(found).length) {
    try {
      localStorage.setItem(KEY, JSON.stringify({ at: Date.now(), attribution: found }));
    } catch {
      /* private mode */
    }
  }
  return found;
}

export function storedAttribution(now = Date.now()): Attribution | null {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "null") as { at?: number; attribution?: Attribution } | null;
    if (raw && typeof raw.at === "number" && now - raw.at < KEEP_MS && raw.attribution && typeof raw.attribution === "object") {
      return raw.attribution;
    }
  } catch {
    /* fall through */
  }
  return null;
}

export type FunnelEvent = "landing_visit" | "primary_cta_click" | "signup_started";

/** Fire-and-forget; survives the page navigating away (keepalive). */
export function track(event: FunnelEvent, surface: string): void {
  try {
    void fetch("/api/public/v1/events", {
      method: "POST",
      keepalive: true,
      credentials: "omit",
      headers: { "content-type": "application/json", "x-request-id": newRequestId() },
      body: JSON.stringify({ event, surface, attribution: storedAttribution() ?? captureAttribution() }),
    }).catch(() => undefined);
  } catch {
    /* never block */
  }
}
