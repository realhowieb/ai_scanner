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
};

type EventLike = { event_id?: string | null; id?: number | null; ticker?: string | null; message: string; fired_at?: string | null };

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
    };
    meta.set(item, { t, breakout });
    out.push(item);
  }
  return out.map((item) => ({ ...item, ticker: item.rows[0]?.ticker ?? item.ticker }));
}
