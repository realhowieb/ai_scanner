export function pct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

/** PreBreakout probability. The API sends it in percent (0–100), as PreBreakoutProb% is stored. */
export function probPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${Math.round(v)}%`;
}

/** PreBreakout % plus where the name's raw model score sits in the scan, e.g. "13% · top 5%".
 * Calibration gives many names the same %, so the rank is what tells them apart. */
export function probWithRank(v: number | null | undefined, rank: number | null | undefined): string {
  const p = probPct(v);
  if (p === "—" || rank === null || rank === undefined || !Number.isFinite(rank)) return p;
  return `${p} · top ${Math.round(rank)}%`;
}

export function price(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: v < 1 ? 4 : 2 })}`;
}

export function num(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function compact(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

/** "Oct 6, 9:42 AM ET". Times from the API always carry a timezone. */
export function etTime(iso: string | null | undefined, withDate = true): string {
  if (!iso) return "time unavailable";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "time unavailable";
  const opts: Intl.DateTimeFormatOptions = { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" };
  if (withDate) Object.assign(opts, { month: "short", day: "numeric" });
  return `${d.toLocaleString("en-US", opts)} ET`;
}

export function etDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { timeZone: "America/New_York", weekday: "long", month: "long", day: "numeric" });
}

/** "Oct 8" (ET), or "Oct 8, 2025" outside the current year. */
export function shortDate(iso: string | null | undefined, nowMs = Date.now()): string {
  if (!iso) return "";
  const d = new Date(iso.length === 10 ? `${iso}T12:00:00Z` : iso);
  if (Number.isNaN(d.getTime())) return "";
  const year = (x: Date) => x.toLocaleDateString("en-US", { timeZone: "America/New_York", year: "numeric" });
  return d.toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", ...(year(d) !== year(new Date(nowMs)) ? { year: "numeric" } : {}) });
}

export function minutesSince(iso: string | null | undefined, nowMs = Date.now()): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : Math.max(0, Math.round((nowMs - t) / 60000));
}

/** Scans run several times a day; older than this reads "stale" (the web's 6 hours). */
export const STALE_AFTER_MIN = 360;

/** "8h ago"; null when the time is unknown. */
export function ago(iso: string | null | undefined, nowMs = Date.now()): string | null {
  const m = minutesSince(iso, nowMs);
  if (m === null) return null;
  return m < 90 ? `${m}m ago` : m < 36 * 60 ? `${Math.floor(m / 60)}h ago` : `${Math.floor(m / 1440)}d ago`;
}

export function freshness(iso: string | null | undefined, nowMs = Date.now()): { label: string; stale: boolean } {
  const a = ago(iso, nowMs);
  if (a === null) return { label: "Updated time unavailable", stale: true };
  return { label: `Updated ${a}`, stale: (minutesSince(iso, nowMs) ?? 0) >= STALE_AFTER_MIN };
}

export function greeting(iso: string, nowMs = Date.now()): string {
  void nowMs;
  const h = Number(new Date(iso).toLocaleString("en-US", { timeZone: "America/New_York", hour: "numeric", hour12: false }));
  return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

export const SETUP_LABELS: Record<string, string> = {
  breakout: "Breakout",
  golden_cross: "Golden cross",
  prebreakout: "PreBreakout",
  gapper: "Gapper",
  gainer: "Gainer",
};

export function setupLabel(s: string | null | undefined): string {
  if (!s) return "Signal";
  return SETUP_LABELS[s] || s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
