// Fired-alert messages are stored as text ("Breakout alert: NVAX: BreakoutScore 62.2 (≥ 50)\nXP: …\n…and 19 more.").
// This turns them into something the Recently fired list can lay out: one header per alert,
// tickers sorted strongest first, and same-scan breakout alerts at different thresholds merged.

export type FiredRow = { ticker: string; value: number | null; detail: string; earnings: string | null };

export type FiredItem = {
  key: string;
  firedAt: string | null | undefined;
  /** "Breakout", "Watchlist", "Move", or the message itself when it isn't in the usual shape. */
  title: string;
  /** Tickers strongest first (by value when every row has one). */
  rows: FiredRow[];
  /** Names matched in total, including ones cut from the stored message ("…and N more"). */
  total: number;
  /** Per-threshold counts when breakout alerts at different thresholds fired on the same scan. */
  tiers: { rule: string; count: number }[];
  /** Free text when the message isn't a list of ticker lines (e.g. a move alert). */
  text: string | null;
  ticker: string | null;
  /** The saved alerts / rules behind it ("alert:<alert id>", "rule:<rule id>"), for per-rule counts. */
  sources: string[];
  /** Tickers that weren't in the previous firing of the same kind of alert; null when there's no earlier one loaded. */
  fresh: Set<string> | null;
};

type EventLike = {
  event_id?: string | null; id?: number | null; ticker?: string | null; message: string; fired_at?: string | null;
  alert_id?: number | null; rule_id?: number | null;
};

function sourceOf(e: EventLike): string | null {
  if (e.rule_id != null) return `rule:${e.rule_id}`;
  if (e.alert_id != null) return `alert:${e.alert_id}`;
  return null;
}

const MORE = /^…and (\d+) more\.?$/;
const RULE = /\(≥ ([0-9.]+)\)/;
const EARNINGS = /\s*⚠️\s*(earnings in \d+d)\s*$/;
const LINE = /^([A-Z0-9.\-]{1,10}):\s*(.*)$/;
// Breakout alerts that fired within this window are the same scan run.
const SAME_SCAN_MS = 15 * 60 * 1000;

type Parsed = { label: string; rule: string | null; rows: FiredRow[]; extra: number; text: string | null };

export function parseMessage(message: string): Parsed {
  const msg = String(message ?? "");
  const idx = msg.indexOf(": ");
  const label = idx > 0 && msg.slice(0, idx).endsWith("alert") ? msg.slice(0, idx) : "";
  const rest = label ? msg.slice(idx + 2) : msg;
  const rows: FiredRow[] = [];
  let extra = 0;
  let rule: string | null = null;
  const leftovers: string[] = [];
  for (const raw of rest.split("\n")) {
    const ln = raw.trim();
    if (!ln) continue;
    const more = MORE.exec(ln);
    if (more) { extra += Number(more[1]); continue; }
    const m = LINE.exec(ln);
    if (!m) { leftovers.push(ln); continue; }
    let detail = m[2]!;
    const earn = EARNINGS.exec(detail);
    if (earn) detail = detail.slice(0, earn.index);
    const r = RULE.exec(detail);
    if (r && rule === null) rule = r[1]!;
    detail = detail.replace(RULE, "").trim();
    const num = /BreakoutScore (-?[0-9.]+)/.exec(detail);
    rows.push({ ticker: m[1]!, value: num ? Number(num[1]) : null, detail, earnings: earn ? earn[1]! : null });
  }
  // A message that isn't ticker lines ("Move alert: NVDA -2.8% today …") keeps its text.
  if (rows.length === 0) return { label, rule: null, rows: [], extra: 0, text: rest.trim() || msg };
  return { label, rule, rows, extra, text: leftovers.length ? leftovers.join(" ") : null };
}

function shortTitle(label: string): string {
  return label.replace(/ alert$/, "") || "Alert";
}

function byStrength(rows: FiredRow[]): FiredRow[] {
  if (rows.some((r) => r.value === null)) return rows;
  return [...rows].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
}

function ms(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? null : t;
}

/** Events newest first in, display items newest first out. */
export function groupFired(events: EventLike[]): FiredItem[] {
  const out: FiredItem[] = [];
  const meta = new Map<FiredItem, { t: number | null; breakout: boolean }>();
  for (const e of events) {
    const p = parseMessage(e.message);
    const key = e.event_id ?? String(e.id ?? out.length);
    const t = ms(e.fired_at);
    const breakout = p.label === "Breakout alert" && p.rows.length > 0;
    if (breakout) {
      const prev = out.find((o) => {
        const m = meta.get(o)!;
        return m.breakout && m.t !== null && t !== null && Math.abs(m.t - t) <= SAME_SCAN_MS
          && !o.tiers.some((x) => x.rule === (p.rule ?? ""));
      });
      if (prev) {
        const seen = new Map(prev.rows.map((r) => [r.ticker, r]));
        for (const r of p.rows) if (!seen.has(r.ticker)) seen.set(r.ticker, r);
        prev.rows = byStrength([...seen.values()]);
        prev.tiers.push({ rule: p.rule ?? "", count: p.rows.length + p.extra });
        prev.tiers.sort((a, b) => Number(b.rule) - Number(a.rule));
        prev.total = Math.max(prev.total, prev.rows.length, p.rows.length + p.extra);
        const src = sourceOf(e);
        if (src && !prev.sources.includes(src)) prev.sources.push(src);
        continue;
      }
    }
    const rows = byStrength(p.rows);
    const item: FiredItem = {
      key,
      firedAt: e.fired_at,
      title: shortTitle(p.label),
      rows,
      total: rows.length + p.extra,
      tiers: breakout ? [{ rule: p.rule ?? "", count: rows.length + p.extra }] : [],
      text: p.text,
      ticker: rows[0]?.ticker ?? e.ticker ?? null,
      sources: [sourceOf(e)].filter((x): x is string => !!x),
      fresh: null,
    };
    meta.set(item, { t, breakout });
    out.push(item);
  }
  const items = out.map((item) => ({ ...item, ticker: item.rows[0]?.ticker ?? item.ticker }));
  // Oldest first: each list alert compared with the one before it of the same kind.
  const last = new Map<string, Set<string>>();
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i]!;
    if (item.rows.length === 0) continue;
    const names = new Set(item.rows.map((r) => r.ticker));
    const prev = last.get(item.title);
    item.fresh = prev ? new Set([...names].filter((t) => !prev.has(t))) : null;
    last.set(item.title, names);
  }
  return items;
}

/** Calendar day in New York, for grouping ("2026-11-17"). */
export function etDay(iso: string | null | undefined): string {
  const t = ms(iso);
  if (t === null) return "";
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(t));
}

/** "Mon Nov 17" for a day from etDay. */
export function dayLabel(day: string): string {
  if (!day) return "Unknown day";
  const d = new Date(`${day}T12:00:00Z`);
  return new Intl.DateTimeFormat("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" }).format(d);
}

/** The best number on an alert: its top score, or nothing when the rows carry none. */
export function bestValue(item: FiredItem): number | null {
  const vals = item.rows.map((r) => r.value).filter((v): v is number => v !== null);
  return vals.length ? Math.max(...vals) : null;
}

export type TickerStat = { ticker: string; count: number; kinds: string[]; first: string | null; last: string | null; best: number | null };

/** Per ticker across the given alerts (newest first in): times it fired, kinds, first/last time, best score. */
export function tickerStats(items: FiredItem[]): TickerStat[] {
  const by = new Map<string, TickerStat>();
  for (const item of items) {
    const names = item.rows.length ? item.rows : item.ticker ? [{ ticker: item.ticker, value: null }] : [];
    for (const r of names) {
      const s = by.get(r.ticker) ?? { ticker: r.ticker, count: 0, kinds: [], first: null, last: null, best: null };
      s.count += 1;
      if (!s.kinds.includes(item.title)) s.kinds.push(item.title);
      const at = item.firedAt ?? null;
      if (at && (!s.last || ms(at)! > ms(s.last)!)) s.last = at;
      if (at && (!s.first || ms(at)! < ms(s.first)!)) s.first = at;
      if (r.value !== null && (s.best === null || r.value > s.best)) s.best = r.value;
      by.set(r.ticker, s);
    }
  }
  return [...by.values()].sort((a, b) => b.count - a.count || (b.best ?? -1) - (a.best ?? -1) || a.ticker.localeCompare(b.ticker));
}
