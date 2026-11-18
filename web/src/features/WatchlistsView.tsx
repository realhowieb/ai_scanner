"use client";

// Watchlists: list, create, rename, make default, delete; tickers with notes. Every
// change is shown only after the server confirms it (then the affected data reloads).
// Scores and prices come from the latest market scan, labelled with its time: the API
// attaches each ticker's scan row (items[].latest); an older API without it falls back to
// matching the plan's ranked rows. Tickers that aren't ranked setups show no score
// rather than a made-up quote. GET /v1/watchlists/{id}/intelligence adds what changed since
// the previous scan (score, rank), PreBreakout (Premium), RVOL and alert counts; all of it
// is computed by the API, and an API without that route just leaves those facts out.
// GET /v1/quotes gives each ticker's latest trade (Alpaca, shared about a minute), shown in
// place of the scan's price; without it the scan price or the daily bar is shown, labelled.
import Link from "@/components/Link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { alertRules, parseTickers, watchlists } from "@/api/userData";
import type { Watchlist, WatchlistDetail, WatchlistIntelItem } from "@/api/userData";
import { ConfirmDialog, Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Freshness, Pill, RefreshStatus, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { useNarrow } from "@/hooks/useNarrow";
import { etTime, num, pct, price, setupLabel, shortDate } from "@/lib/format";
import { tabKey } from "@/lib/tabs";

import { csvFile, ResearchCsvButton, round } from "./ResearchCsv";
import type { CsvTable } from "./ResearchCsv";
import { DailySpark, useDailyBarsState } from "./Sparks";
import { StarterWatchlistButton } from "./Starters";
import { WatchlistAINote } from "./WatchlistAINote";
import { WatchlistPerformanceCard } from "./WatchlistPerformance";
import { AutoTag } from "./SetupTable";

const NOTE_MAX = 500;

/** Summary chip filters: each chip lists the tickers it counts. */
type WlFilter = "all" | "scan" | "auto" | "attention" | "notes";

/** Why a ticker counts as "needs attention": Fading in the scan, or its HSF Score fell since the previous scan. */
export function attentionReasons(fading: boolean | null | undefined, scoreChange: number | null | undefined): string[] {
  const out: string[] = [];
  if (fading) out.push("Fading");
  if ((scoreChange ?? 0) < 0) out.push(`Score −${Math.abs(scoreChange!)} since last scan`);
  return out;
}
/** GET /v1/quotes answers at most this many tickers per request. */
export const QUOTE_CHUNK = 50;
type LatestQuotes = Schemas["LatestQuotes"];

/** Latest trades for every ticker: one /v1/quotes request per 50 (sorted, so each piece is
 * shared by the API's quote cache), merged. A piece that fails leaves its tickers on the scan
 * price, labelled as such; only when every piece fails is the error shown. */
export async function watchlistQuotes(tickers: string[], signal: AbortSignal): Promise<LatestQuotes> {
  const pieces: string[][] = [];
  for (let i = 0; i < tickers.length; i += QUOTE_CHUNK) pieces.push(tickers.slice(i, i + QUOTE_CHUNK));
  const results = await Promise.allSettled(pieces.map((p) =>
    unwrap(api.GET("/v1/quotes", { params: { query: { symbols: p.join(",") } }, signal }))));
  const ok = results.filter((r): r is PromiseFulfilledResult<LatestQuotes> => r.status === "fulfilled");
  if (!ok.length) throw (results[0] as PromiseRejectedResult).reason;
  return { quotes: Object.assign({}, ...ok.map((r) => r.value.quotes ?? {})), feed: ok.find((r) => r.value.feed)?.value.feed ?? null };
}
const tone = (v: number | null | undefined) => ((v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "");
const metric = (v: unknown, fallback = 0) => (typeof v === "number" || typeof v === "string" ? v : fallback);

type NameProps = {
  title: string; initial?: string; withDefault?: boolean; submitLabel: string; error?: ReactNode;
  onClose: () => void; onSubmit: (name: string, makeDefault: boolean) => Promise<boolean>;
};

function NameDialog({ open, ...p }: NameProps & { open: boolean }) {
  // The form mounts with the dialog, so each opening starts from `initial`.
  return <Dialog open={open} title={p.title} onClose={p.onClose}><NameForm {...p} /></Dialog>;
}

function NameForm({ initial = "", withDefault, submitLabel, error, onClose, onSubmit }: NameProps) {
  const [name, setName] = useState(initial);
  const [makeDefault, setMakeDefault] = useState(false);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    const ok = await onSubmit(name.trim(), makeDefault);
    setBusy(false);
    if (ok) onClose();
  };
  return (
    <form className="stack-sm" onSubmit={submit}>
      <label className="field"><span>Name</span>
        <input data-autofocus value={name} maxLength={80} required onChange={(e) => setName(e.target.value)} />
      </label>
      {withDefault && (
        <label className="check"><input type="checkbox" checked={makeDefault} onChange={(e) => setMakeDefault(e.target.checked)} />
          <span>Make it my default watchlist</span></label>
      )}
      {error}
      <div className="row-actions end">
        <button type="button" className="btn" onClick={onClose}>Cancel</button>
        <button type="submit" className="btn btn-primary" disabled={busy || !name.trim()}>{busy ? "Saving…" : submitLabel}</button>
      </div>
    </form>
  );
}

function NoteEditor({ wl, ticker, note, onSaved, startOpen = false, onClose }: {
  wl: number; ticker: string; note: string | null | undefined; onSaved: () => void; startOpen?: boolean; onClose?: () => void;
}) {
  const [editing, setEditingState] = useState(startOpen);
  const setEditing = (v: boolean) => { setEditingState(v); if (!v) onClose?.(); };
  const [text, setText] = useState(note ?? "");
  const act = useAction();
  const save = async (e: FormEvent) => {
    e.preventDefault();
    const ok = await act.run(async () => { await watchlists.setNote(wl, ticker, text); return true; });
    if (ok) {
      setEditing(false);
      onSaved();
    }
  };
  if (!editing) {
    return (
      <div className="note">
        {note ? <span className="body-sm">{note}</span> : <span className="cap">No note</span>}
        <button type="button" className="link-btn" onClick={() => { setText(note ?? ""); setEditing(true); }} aria-label={`${note ? "Edit" : "Add"} note for ${ticker}`}>
          {note ? "Edit" : "Add note"}
        </button>
      </div>
    );
  }
  return (
    <form className="stack-sm" onSubmit={save}>
      <label className="sr-only" htmlFor={`note-${ticker}`}>Note for {ticker}</label>
      <textarea id={`note-${ticker}`} className="textarea" rows={2} maxLength={NOTE_MAX} value={text} autoFocus
        onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === "Escape") setEditing(false); }} />
      <div className="row-actions">
        <button type="submit" className="btn btn-sm btn-primary" disabled={act.busy}>{act.busy ? "Saving…" : "Save note"}</button>
        <button type="button" className="btn btn-sm" onClick={() => setEditing(false)}>Cancel</button>
        <span className="cap">{text.length}/{NOTE_MAX}</span>
      </div>
      <ErrorLine error={act.error} />
    </form>
  );
}

function AddTickers({ wl, onAdded }: { wl: number; onAdded: () => void }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const act = useAction();
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const tickers = parseTickers(text);
    if (!tickers.length) return;
    const r = await act.run(() => watchlists.addTickers(wl, tickers));
    if (!r) return;
    const parts = [];
    if (r.added.length) parts.push(`Added ${r.added.join(", ")}.`);
    if (r.already_present.length) parts.push(`Already in the list: ${r.already_present.join(", ")}.`);
    if (r.invalid.length) parts.push(`Not valid ticker symbols (not saved): ${r.invalid.join(", ")}.`);
    setResult(parts.join(" "));
    setText(r.invalid.join(" "));
    if (r.added.length) onAdded();
  };
  return (
    <form className="stack-sm" onSubmit={submit}>
      <div className="inline-form">
        <label className="field grow"><span>Add tickers</span>
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="AAPL, MSFT NVDA" autoComplete="off" />
        </label>
        <button type="submit" className="btn btn-primary" disabled={act.busy || !text.trim()}>{act.busy ? "Adding…" : "Add"}</button>
      </div>
      <p className="cap">Separate with spaces or commas, up to 200 at a time.</p>
      {result && <p className="notice" role="status">{result}</p>}
      <ErrorLine error={act.error} />
    </form>
  );
}

/** What the server's Watchlist Intelligence adds to a ticker row. */
function IntelFacts({ x }: { x: WatchlistIntelItem }) {
  const sc = x.score_change ?? 0;
  const rc = x.rank_change ?? 0;
  const alerts = x.active_alert_count ?? 0;
  return (
    <>
      {sc !== 0 && <span className={`delta-chip mono ${sc > 0 ? "up" : "down"}`} title="HSF Score change since the previous scan">{sc > 0 ? "▲ +" : "▼ −"}{Math.abs(sc)} HSF</span>}
      {rc !== 0 && <span>{rc > 0 ? "Up" : "Down"} {Math.abs(rc)} place{Math.abs(rc) === 1 ? "" : "s"}</span>}
      {x.prebreakout && <Pill tone="gold">PreBreakout</Pill>}
      {x.rvol != null && <span>RVOL {num(x.rvol)}x</span>}
      {alerts > 0 && <span>{alerts} alert{alerts === 1 ? "" : "s"}</span>}
    </>
  );
}

/** Alert rules on this watchlist: the server evaluates them after each market scan. */
function WatchlistRules({ wl, onChanged }: { wl: number; onChanged: () => void }) {
  const rules = useApi("alert-rules", (signal) => alertRules.list(signal));
  const types = useApi("alert-rule-types", (signal) => alertRules.types(signal));
  const [type, setType] = useState("");
  const [threshold, setThreshold] = useState("");
  const add = useAction();
  const rm = useAction();
  const failed = rules.error ?? types.error;
  if (failed && failed.status === 404) return null; // an API without rules: no card
  if (failed && !(rules.data && types.data)) {
    return <Card title="Alerts on this list" id="wl-rules"><ErrorState error={failed} onRetry={() => { rules.reload(); types.reload(); }} what="alerts on this list" /></Card>;
  }
  if (!rules.data || !types.data) return null;
  const available = types.data.filter((t) => t.available && t.scope !== "market");
  const spec = available.find((t) => t.type === type) ?? available[0];
  if (!spec) return null;
  const mine = rules.data.rules.filter((r) => r.watchlist_id === wl);
  const label = (t: string) => types.data?.find((x) => x.type === t)?.label ?? t;
  const changed = () => { rules.reload(); onChanged(); };
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const value = spec.threshold ? Number(threshold || spec.threshold.default) : undefined;
    const ok = await add.run(async () => {
      await alertRules.create({ rule_type: spec.type, watchlist_id: wl, enabled: true, ...(value !== undefined ? { threshold: value } : {}) });
      return true;
    });
    if (ok) { setThreshold(""); changed(); }
  };
  return (
    <Card title="Alerts on this list" id="wl-rules" aside={`${rules.data.used} of ${rules.data.limit} alerts in use`}>
      {mine.length > 0 && (
        <ul className="stack-sm">
          {mine.map((r) => (
            <li key={r.id} className="row-actions">
              <span>{label(r.rule_type)}{r.threshold != null ? ` ${r.threshold}` : ""}</span>
              {!r.enabled && <Pill tone="warn">Off</Pill>}
              <button type="button" className="link-btn" disabled={rm.busy} aria-label={`Delete alert ${label(r.rule_type)}`}
                onClick={() => void rm.run(async () => { await alertRules.remove(r.id); changed(); return true; })}>Delete</button>
            </li>
          ))}
        </ul>
      )}
      <form className="inline-form" onSubmit={submit}>
        <label className="field"><span>Alert me when any ticker</span>
          <select value={spec.type} onChange={(e) => { setType(e.target.value); setThreshold(""); }}>
            {available.map((t) => <option key={t.type} value={t.type}>{t.label}</option>)}
          </select>
        </label>
        {spec.threshold && (
          <label className="field"><span>Value</span>
            <input type="number" inputMode="decimal" min={spec.threshold.min} max={spec.threshold.max} step="any"
              placeholder={String(spec.threshold.default ?? "")} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
          </label>
        )}
        <button type="submit" className="btn" disabled={add.busy}>{add.busy ? "Adding…" : "Add alert"}</button>
      </form>
      <ErrorLine error={add.error ?? rm.error} />
      <p className="cap">{spec.description} Checked after each market scan; fired alerts show on <Link href="/alerts">Alerts</Link>.</p>
    </Card>
  );
}

/** What the price column is. A trade's own time is shown per row: on a weekend or holiday the
 * latest trade can be days old even though the API re-reads it about every minute. */
export function priceNote(hasQuotes: boolean, loading: boolean, feed = ""): string {
  if (hasQuotes) return `Prices are each ticker's latest trade, with its time; HSF re-reads them about every minute.${feed} Scores change when a new scan runs.`;
  if (loading) return "Loading the latest trades. Scores change when a new scan runs.";
  return "Latest trades aren't available right now, so prices are from the scan or the daily bar, as labelled on each row. Scores change when a new scan runs.";
}

type Change = Schemas["WatchlistChange"];
const titleCase = (v: string) => v.charAt(0) + v.slice(1).toLowerCase();

/** One readable chip per change the count includes, e.g. "BRZE score 88 → 76" and "BRZE Strong → Watch". */
function describeChange(c: Change): string {
  const from = c.previous_score, to = c.current_score;
  const scores = from != null && to != null ? ` ${Math.round(from)} → ${Math.round(to)}` : "";
  switch (c.event_type) {
    case "NEW_OPPORTUNITY": return `${c.ticker} newly ranked${to != null ? ` at ${Math.round(to)}` : ""}`;
    case "DROPPED": return `${c.ticker} left the ranked list`;
    case "RISING": return `${c.ticker} score${scores || " rising"}`;
    case "FALLING": return `${c.ticker} score${scores || " falling"}`;
    case "FADING": return `${c.ticker} fading`;
    case "STATUS_UPGRADE":
    case "STATUS_DOWNGRADE":
      return c.previous_status && c.current_status ? `${c.ticker} ${titleCase(c.previous_status)} → ${titleCase(c.current_status)}`
        : `${c.ticker} status ${c.event_type === "STATUS_UPGRADE" ? "up" : "down"}`;
    case "SIGNAL_ADDED": return `${c.ticker} new signal${c.signal ? `: ${c.signal.replace(/_/g, " ")}` : ""}`;
    case "SIGNAL_REMOVED": return `${c.ticker} lost signal${c.signal ? `: ${c.signal.replace(/_/g, " ")}` : ""}`;
    default: return `${c.ticker} ${c.event_type.replace(/_/g, " ").toLowerCase()}`;
  }
}

function changeTone(t: string): "up" | "down" | undefined {
  if (t === "NEW_OPPORTUNITY" || t === "RISING" || t === "STATUS_UPGRADE" || t === "SIGNAL_ADDED") return "up";
  if (t === "DROPPED" || t === "FALLING" || t === "STATUS_DOWNGRADE" || t === "SIGNAL_REMOVED" || t === "FADING") return "down";
  return undefined;
}

function WatchlistChangesCard({ id }: { id: number }) {
  const changes = useApi(`wl-changes:${id}`, (signal) => watchlists.changes(id, signal));
  if (changes.error || !changes.data) return null;
  const data = changes.data;
  const s = data.summary ?? {};
  if (data.changes.length === 0 && data.alerts.length === 0 && data.headline.length === 0 && !Number(s.total_changes)) {
    return <p className="cap watchlist-quiet" role="status">{data.has_baseline ? "No major watchlist changes since the previous scan." : "Changes appear after two scans include this list."}</p>;
  }
  return (
    <Card title="Since last scan" id="wl-changes" aside={data.has_baseline ? undefined : "Waiting for a baseline"}>
      <div className="metric-strip" aria-label="Watchlist changes summary">
        <span><strong>{metric(s.total_changes, data.changes.length)}</strong> changes</span>
        <span><strong>{metric(s.new, 0)}</strong> new</span>
        <span><strong>{metric(s.rising, 0)}</strong> rising</span>
        <span><strong>{metric(s.falling, 0)}</strong> falling</span>
        <span><strong>{metric(s.alerts, data.alerts.length)}</strong> alerts</span>
      </div>
      {data.changes.length ? (
        <div className="chips">
          {data.changes.slice(0, 12).map((c, i) => <Pill key={`${c.ticker}-${c.event_type}-${i}`} tone={changeTone(c.event_type)}>{describeChange(c)}</Pill>)}
          {data.changes.length > 12 && <span className="cap">+{data.changes.length - 12} more</span>}
        </div>
      ) : data.headline.length ? (
        <div className="chips">
          {data.headline.slice(0, 8).map((h) => <Pill key={`${h.ticker}-${h.event_type}`}>{h.ticker} {h.event_type.replace(/_/g, " ").toLowerCase()}</Pill>)}
        </div>
      ) : <p className="cap">{data.has_baseline ? "No major watchlist changes since the previous scan." : "Changes appear after two scans include this list."}</p>}
    </Card>
  );
}

type CsvQuote = { last?: number | null; chg_pct?: number | null; at?: string | null; source?: string | null };
type CsvScore = { score?: number | null; primary_setup?: string | null; last?: number | null; chg_pct?: number | null; rank?: number | null };

/** Premium CSV of a watchlist: what the Tickers card shows, one row per ticker. */
export function watchlistCsv(items: WatchlistDetail["items"], scores: Map<string, CsvScore>, quotes: Record<string, CsvQuote>, scanAt?: string | null): CsvTable {
  return {
    header: ["Ticker", "HSF Score", "Setup", "Rank in scan", "Price", "Change %", "Price source", "Added", "Price when added", "Note", "Scores from scan (UTC)"],
    rows: items.map((it) => {
      const s = scores.get(it.ticker);
      const q = quotes[it.ticker];
      const priceNow = q?.last ?? s?.last ?? null;
      return [it.ticker, s?.score ?? null, s ? setupLabel(s.primary_setup) : "", it.latest?.rank ?? null,
        round(priceNow, priceNow != null && Math.abs(priceNow) < 1 ? 4 : 2), round(q ? q.chg_pct : s?.chg_pct),
        q ? (q.source === "daily_bar" ? "daily close" : "latest trade") : s ? "scan" : "",
        it.added_at ?? "", round(it.price_when_added), it.note ?? "", scanAt ?? ""];
    }),
  };
}


export type SortKey = "ticker" | "score" | "price" | "chg" | "since" | "rank" | "rvol" | "added";
type SortState = { key: SortKey; desc: boolean };
type ViewPrefs = { sort: SortState; view: "table" | "cards" };
const PREFS_KEY = "hsf.watchlists.v1";
const DEFAULT_PREFS: ViewPrefs = { sort: { key: "score", desc: true }, view: "table" };
const SORT_LABEL: Record<SortKey, string> = {
  ticker: "Ticker", score: "HSF Score", price: "Price", chg: "Today", since: "Since added", rank: "Rank in scan", rvol: "RVOL", added: "Date added",
};
/** Ticker and rank read best smallest first; everything else biggest first. */
const ASC_FIRST: SortKey[] = ["ticker", "rank"];

function readPrefs(): ViewPrefs {
  try {
    const v = JSON.parse(localStorage.getItem(PREFS_KEY) ?? "null") as Partial<ViewPrefs> | null;
    if (v && v.sort && v.sort.key in SORT_LABEL) return { sort: { key: v.sort.key, desc: !!v.sort.desc }, view: v.view === "cards" ? "cards" : "table" };
  } catch { /* private window or blocked storage: defaults */ }
  return DEFAULT_PREFS;
}
function usePrefs(): [ViewPrefs, (p: ViewPrefs) => void] {
  // Detail only mounts once the lists have loaded in the browser, so reading storage here is safe.
  const [prefs, setPrefs] = useState<ViewPrefs>(() => (typeof window === "undefined" ? DEFAULT_PREFS : readPrefs()));
  const save = (p: ViewPrefs) => {
    setPrefs(p);
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* keep it for this visit only */ }
  };
  return [prefs, save];
}

/** Sort rows by one value; rows without a value always go last, then A to Z. */
export function sortRows<T extends { ticker: string }>(rows: T[], value: (r: T) => number | string | null | undefined, desc: boolean): T[] {
  return [...rows].sort((a, b) => {
    const va = value(a), vb = value(b);
    const na = va === null || va === undefined || (typeof va === "number" && Number.isNaN(va));
    const nb = vb === null || vb === undefined || (typeof vb === "number" && Number.isNaN(vb));
    if (na !== nb) return na ? 1 : -1;
    if (!na && !nb && va !== vb) {
      const c = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
      return desc ? -c : c;
    }
    return a.ticker.localeCompare(b.ticker);
  });
}

/** Percent change from `base` to `now`, or null when either is missing. */
export function sinceAdded(base: number | null | undefined, now: number | null | undefined): number | null {
  if (base == null || now == null || !(base > 0)) return null;
  return (now / base - 1) * 100;
}

/** Key for spotting look-alike list names: "July Top List" and "July top list 2" or "… Copy" match. */
export function dupKey(name: string): string {
  return name.toLowerCase().replace(/(\s+copy)?(\s+\(?\d{1,2}\)?|\s*\(\d{1,2}\))?\s*$/, "").replace(/[^a-z0-9]+/g, "");
}
/** Groups of two or more active lists whose names look alike, biggest list first. */
export function lookAlikes(lists: Watchlist[]): Watchlist[][] {
  const by = new Map<string, Watchlist[]>();
  for (const w of lists) {
    if (w.archived) continue;
    const k = dupKey(w.name);
    if (k) by.set(k, [...(by.get(k) ?? []), w]);
  }
  return [...by.values()].filter((g) => g.length > 1)
    .map((g) => [...g].sort((a, b) => Number(b.is_default) - Number(a.is_default) || b.symbol_count - a.symbol_count));
}
const isEarnings = (w: Watchlist) => /earnings/i.test(w.name);

/** Sidebar groups: Pinned (default + pinned), Lists, Earnings (names with "earnings"), Archived. */
export function groupLists(lists: Watchlist[], query = ""): { pinned: Watchlist[]; active: Watchlist[]; earnings: Watchlist[]; archived: Watchlist[] } {
  const q = query.trim().toLowerCase();
  const shown = q ? lists.filter((w) => w.name.toLowerCase().includes(q)) : lists;
  const pinned = shown.filter((w) => !w.archived && (w.is_default || w.pinned))
    .sort((a, b) => Number(b.is_default) - Number(a.is_default));
  const rest = shown.filter((w) => !w.archived && !w.is_default && !w.pinned);
  return { pinned, active: rest.filter((w) => !isEarnings(w)), earnings: rest.filter(isEarnings), archived: shown.filter((w) => w.archived) };
}

type Item = WatchlistDetail["items"][number];
type Tab = "tickers" | "changes" | "performance" | "ai" | "alerts";
const TABS: [Tab, string][] = [["tickers", "Tickers"], ["changes", "Since last scan"], ["performance", "Performance"], ["ai", "AI rundown"], ["alerts", "List alerts"]];

/** Move or copy the selected tickers to another list. */
function CopyDialog({ open, move, from, tickers, lists, onClose, onDone }: {
  open: boolean; move: boolean; from: WatchlistDetail; tickers: string[]; lists: Watchlist[];
  onClose: () => void; onDone: (message: string) => void;
}) {
  const others = lists.filter((w) => w.id !== from.id).sort((a, b) => Number(!!a.archived) - Number(!!b.archived));
  const [to, setTo] = useState<number | "">("");
  const act = useAction();
  const target = others.find((w) => w.id === to) ?? others[0];
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!target) return;
    const r = await act.run(() => watchlists.copyTickers(from.id, target.id, tickers, move));
    if (!r) return;
    const done = move ? r.moved ?? [] : r.copied ?? [];
    const parts = [`${move ? "Moved" : "Copied"} ${done.length} to ${target.name}.`];
    if (r.already_present?.length) parts.push(`${target.name} already had ${r.already_present.join(", ")}${move ? " (removed from here)" : ""}.`);
    onDone(parts.join(" "));
  };
  return (
    <Dialog open={open} title={`${move ? "Move" : "Copy"} ${tickers.length} ticker${tickers.length === 1 ? "" : "s"}`} onClose={() => { act.clear(); onClose(); }}>
      {others.length === 0 ? <p className="cap">Create another watchlist first.</p> : (
        <form className="stack-sm" onSubmit={submit}>
          <p className="cap">{tickers.join(", ")}</p>
          <label className="field"><span>To watchlist</span>
            <select data-autofocus value={target?.id ?? ""} onChange={(e) => setTo(Number(e.target.value))}>
              {others.map((w) => <option key={w.id} value={w.id}>{w.name}{w.archived ? " (archived)" : ""} · {w.symbol_count}</option>)}
            </select>
          </label>
          <p className="cap">Notes, added dates and prices go with them.{move ? ` They're removed from ${from.name}.` : ""}</p>
          <ErrorLine error={act.error} />
          <div className="row-actions end">
            <button type="button" className="btn" onClick={() => { act.clear(); onClose(); }}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={act.busy}>{act.busy ? "Saving…" : move ? "Move" : "Copy"}</button>
          </div>
        </form>
      )}
    </Dialog>
  );
}

/** Fold another list into `target`, then delete it. */
function MergeDialog({ target, lists, initialFrom, onClose, onMerged }: {
  target: Watchlist | null; lists: Watchlist[]; initialFrom: number | null; onClose: () => void; onMerged: (message: string) => void;
}) {
  const others = target ? lists.filter((w) => w.id !== target.id) : [];
  const [from, setFrom] = useState<number | null>(initialFrom);
  const act = useAction();
  const source = others.find((w) => w.id === from) ?? others[0];
  const close = () => { act.clear(); onClose(); };
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!target || !source) return;
    const r = await act.run(() => watchlists.merge(target.id, source.id));
    if (r) onMerged(`Merged ${source.name} into ${target.name}: ${r.merged} new ticker${r.merged === 1 ? "" : "s"} added.`);
  };
  return (
    <Dialog open={!!target} title={`Merge into ${target?.name ?? ""}`} onClose={close}>
      {others.length === 0 ? <p className="cap">There&apos;s no other list to merge.</p> : (
        <form className="stack-sm" onSubmit={submit}>
          <label className="field"><span>Merge this list in</span>
            <select data-autofocus value={source?.id ?? ""} onChange={(e) => setFrom(Number(e.target.value))}>
              {others.map((w) => <option key={w.id} value={w.id}>{w.name}{w.archived ? " (archived)" : ""} · {w.symbol_count}</option>)}
            </select>
          </label>
          {source && <p className="cap">{source.name}&apos;s tickers that aren&apos;t in {target?.name} yet are added, with their notes. Then {source.name} is deleted and its list alerts are switched off. This can&apos;t be undone.</p>}
          <ErrorLine error={act.error} />
          <div className="row-actions end">
            <button type="button" className="btn" onClick={close}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={act.busy || !source}>{act.busy ? "Merging…" : "Merge and delete"}</button>
          </div>
        </form>
      )}
    </Dialog>
  );
}

const NOTE_ICON = <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z" /></svg>;
const X_ICON = <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>;

function Detail({ id, lists, onChanged, onDeleted, onMerge }: {
  id: number; lists: Watchlist[]; onChanged: () => void; onDeleted: () => void; onMerge: () => void;
}) {
  const detail = useApi(`wl:${id}`, (signal) => watchlists.get(id, signal));
  const performance = useApi(`wl-perf:${id}`, (signal) => watchlists.performance(id, signal));
  const intel = useApi(`wl-intel:${id}`, (signal) => watchlists.intelligence(id, signal));
  const changes = useApi(`wl-changes:${id}`, (signal) => watchlists.changes(id, signal));
  const intelBy = useMemo(() => new Map((intel.data?.items ?? []).map((i) => [i.ticker, i])), [intel.data]);
  const perfBy = useMemo(() => new Map((performance.data?.items ?? []).map((i) => [i.ticker, i])), [performance.data]);
  // Only an API that predates items[].latest (no scan_at key at all) needs the ranked rows.
  const legacy = !!detail.data && detail.data.scan_at === undefined;
  const scan = useApi(legacy ? "wl-scan" : null, (signal) => unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: 200 } }, signal })));
  const narrow = useNarrow();
  const [prefs, setPrefs] = usePrefs();
  const [tab, setTab] = useState<Tab>("tickers");
  const [filter, setFilterState] = useState<WlFilter>("all");
  const setFilter = (f: WlFilter) => setFilterState(f === filter ? "all" : f);
  const [selected, setSelected] = useState<string[]>([]);
  const [selecting, setSelecting] = useState(false); // phone: checkboxes show after "Select"
  const [noteOpen, setNoteOpen] = useState<string | null>(null);
  const [copying, setCopying] = useState<"move" | "copy" | null>(null);
  const [bulkMsg, setBulkMsg] = useState<string | null>(null);
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [manageOpen, setManageOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const act = useAction();
  const ren = useAction();
  const del = useAction();
  const rm = useAction();
  const bulk = useAction();
  const reload = () => { detail.reload(); intel.reload(); performance.reload(); changes.reload(); onChanged(); };

  const scores = useMemo(() => {
    // Premium also gets items[].auto: the scheduled scan's score for a ticker the scan didn't rank.
    if (!legacy) return new Map((detail.data?.items ?? []).filter((i) => i.latest || i.auto).map((i) => [i.ticker, (i.latest ?? i.auto)!]));
    return new Map((scan.data?.setups ?? []).map((s) => [s.ticker, s]));
  }, [legacy, detail.data, scan.data]);
  const bars = useDailyBarsState((detail.data?.items ?? []).map((i) => i.ticker));
  const daily = bars.data;
  const tickerKey = [...new Set((detail.data?.items ?? []).map((i) => i.ticker))].sort().join(",");
  const quotes = useApi(tickerKey ? `wl-quotes:${tickerKey}` : null, (signal) => watchlistQuotes(tickerKey.split(","), signal));
  const quoteBy = quotes.data?.quotes ?? {};
  // Refresh re-reads the list (with each ticker's latest scan row), its changes, the latest
  // trades and the daily bars in place. Prices are the provider's latest trade (shared by the
  // API for about a minute); scores change only when a new scan has run.
  const refreshing = detail.refreshing || intel.refreshing || bars.refreshing || quotes.refreshing || (legacy && scan.refreshing);
  const refresh = () => { detail.reload(); intel.reload(); performance.reload(); changes.reload(); bars.reload(); quotes.reload(); if (legacy) scan.reload(); };

  if (detail.error && !detail.data) return <ErrorState error={detail.error} onRetry={detail.reload} what="this watchlist" />;
  if (!detail.data || detail.loading && detail.data.id !== id) return <Skeleton rows={5} label="Loading watchlist" />;
  const w: WatchlistDetail = detail.data;
  const meta = lists.find((l) => l.id === w.id);
  const pinned = !!meta?.pinned, archived = !!meta?.archived;
  const series = daily?.series ?? {};
  const scanAt = legacy ? scan.data?.scan_at : w.scan_at;
  const stale = legacy ? scan.data?.stale : w.stale;
  const reasonsFor = (t: string) => attentionReasons(scores.get(t)?.fading, intelBy.get(t)?.score_change);
  const matches: Record<WlFilter, (i: Item) => boolean> = {
    all: () => true,
    scan: (i) => legacy ? scores.has(i.ticker) : !!i.latest,
    auto: (i) => !legacy && !i.latest && !!i.auto,
    attention: (i) => reasonsFor(i.ticker).length > 0,
    notes: (i) => !!i.note,
  };
  const count = (f: WlFilter) => w.items.filter(matches[f]).length;
  const priceOf = (t: string) => quoteBy[t]?.last ?? scores.get(t)?.last ?? daily?.last_bar?.[t]?.close ?? null;
  const sinceOf = (i: Item) => {
    const p = perfBy.get(i.ticker);
    const live = sinceAdded(p?.base_price ?? i.price_when_added, quoteBy[i.ticker]?.last ?? scores.get(i.ticker)?.last);
    return live ?? p?.change_pct ?? null;
  };
  const value: Record<SortKey, (i: Item) => number | string | null | undefined> = {
    ticker: (i) => i.ticker,
    score: (i) => scores.get(i.ticker)?.score,
    price: (i) => priceOf(i.ticker),
    chg: (i) => quoteBy[i.ticker]?.chg_pct ?? scores.get(i.ticker)?.chg_pct,
    since: sinceOf,
    rank: (i) => (legacy ? undefined : i.latest?.rank),
    rvol: (i) => intelBy.get(i.ticker)?.rvol,
    added: (i) => (i.added_at ? Date.parse(i.added_at) : null),
  };
  const sort = prefs.sort;
  const items = sortRows(w.items.filter(matches[filter]), value[sort.key], sort.desc);
  const sortBy = (key: SortKey) => setPrefs({ ...prefs, sort: key === sort.key ? { key, desc: !sort.desc } : { key, desc: !ASC_FIRST.includes(key) } });
  const chips: { f: WlFilter; n: number; label: string; tip: string }[] = [
    { f: "all", n: w.symbol_count, label: "tracked", tip: "Every ticker on this list. Click to show them all." },
    { f: "scan", n: count("scan"), label: "in latest scan", tip: "Made the latest market scan's ranked list. Their rank is shown on the row." },
    ...(!legacy && count("auto") > 0 ? [{ f: "auto" as const, n: count("auto"), label: "auto-scored", tip: "Not in the latest scan's ranked list, so Premium scored them after the scan (the A mark)." }] : []),
    { f: "attention", n: count("attention"), label: "need attention", tip: "Fading, or the HSF Score fell since the previous scan. The reason is on each row." },
    { f: "notes", n: count("notes"), label: "with notes", tip: "Tickers you've written a note on." },
  ];
  const filterLabel = chips.find((c) => c.f === filter)?.label;
  const hasQuotes = Object.keys(quoteBy).length > 0;
  const feed = quotes.data?.feed === "iex" ? " Prices come from one exchange (IEX), so they can differ slightly from the consolidated price." : "";
  const view = narrow ? "rows" : prefs.view;
  const showChecks = !narrow || selecting;
  const sel = selected.filter((t) => w.items.some((i) => i.ticker === t));
  const allShown = items.length > 0 && items.every((i) => sel.includes(i.ticker));
  const toggle = (t: string) => setSelected((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  const toggleAll = () => setSelected(allShown ? sel.filter((t) => !items.some((i) => i.ticker === t)) : [...new Set([...sel, ...items.map((i) => i.ticker)])]);
  const changeCount = Number(changes.data?.summary?.total_changes ?? changes.data?.changes.length ?? 0) || 0;

  // One price cell: the latest trade (time in the label), else the scan's price, else the daily bar.
  const priceCell = (it: Item) => {
    const q = quoteBy[it.ticker], s = scores.get(it.ticker), bar = daily?.last_bar?.[it.ticker];
    if (q) {
      const label = q.source === "daily_bar" ? `daily close${q.at ? ` ${shortDate(q.at)}` : ""} · no recent trade` : q.at ? `last trade ${etTime(q.at)}` : "latest trade";
      return <span className="mono" title={label}>{price(q.last)}{q.source === "daily_bar" && <span className="wl-src" aria-hidden="true"> close</span>}<span className="sr-only">{label}</span></span>;
    }
    if (s) return <span className="mono" title="Price at the scan">{price(s.last)}<span className="wl-src"> scan</span></span>;
    if (bar) return <span className="mono" title="May be in progress; not live">{price(bar.close)} <span className="wl-src">daily bar {shortDate(bar.session)}</span></span>;
    return <span className="cap">—</span>;
  };
  const chgOf = (t: string) => quoteBy[t]?.chg_pct ?? scores.get(t)?.chg_pct;
  const removeBtn = (t: string) => (
    <button type="button" className="icon-btn" aria-label={`Remove ${t} from ${w.name}`} onClick={() => { rm.clear(); setRemoving(t); }}>{X_ICON}</button>
  );
  const noteBtn = (it: Item) => (
    <button type="button" className={`wl-note-btn${it.note ? " has" : ""}`} aria-label={noteOpen === it.ticker ? `Close the ${it.ticker} note editor` : `${it.note ? "Edit" : "Add"} note for ${it.ticker}`}
      aria-expanded={noteOpen === it.ticker} onClick={() => setNoteOpen(noteOpen === it.ticker ? null : it.ticker)} title={it.note ?? "Add a note"}>
      {NOTE_ICON}{it.note && <span className="wl-note-text">{it.note}</span>}
    </button>
  );
  const noteEditor = (it: Item) => (
    <NoteEditor key={`${it.ticker}-${it.note ?? ""}`} wl={w.id} ticker={it.ticker} note={it.note} startOpen
      onClose={() => setNoteOpen(null)} onSaved={detail.reload} />
  );
  const scoreDelta = (t: string) => {
    const sc = intelBy.get(t)?.score_change ?? 0;
    return sc !== 0 ? <span className={`wl-delta mono ${sc > 0 ? "up" : "down"}`} title="HSF Score change since the previous scan">{sc > 0 ? "▲" : "▼"}{Math.abs(sc)}</span> : null;
  };
  const th = (key: SortKey, label: ReactNode, cls = "") => (
    <th scope="col" className={cls} aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      <button type="button" className="th-sort" onClick={() => sortBy(key)} aria-label={`Sort by ${SORT_LABEL[key]}`}>
        {label}{sort.key === key && <span aria-hidden="true" className="th-arrow">{sort.desc ? " ▼" : " ▲"}</span>}
      </button>
    </th>
  );
  const COLS = 12;

  const table = (
    <div className="table-wrap" role="region" aria-label={`${w.name} tickers`} tabIndex={0}>
      <table className="table wl-table">
        <thead>
          <tr>
            <th scope="col" className="wl-check"><input type="checkbox" aria-label="Select all shown tickers" checked={allShown} onChange={toggleAll} /></th>
            {th("ticker", "Ticker")}
            {th("score", "HSF")}
            <th scope="col">Setup</th>
            {th("price", "Price", "num")}
            {th("chg", "Today", "num")}
            {th("since", "Since added", "num")}
            {th("rank", "Rank", "num")}
            {th("rvol", "RVOL", "num wl-mid")}
            <th scope="col" className="wl-wide">30 days</th>
            <th scope="col">Note</th>
            <th scope="col"><span className="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {items.map((it) => {
            const s = scores.get(it.ticker);
            const x = intelBy.get(it.ticker);
            const rank = !legacy ? it.latest?.rank : undefined;
            const why = reasonsFor(it.ticker);
            const since = sinceOf(it);
            const alerts = x?.active_alert_count ?? 0;
            const rc = x?.rank_change ?? 0;
            return [
              <tr key={it.ticker} className={`${sel.includes(it.ticker) ? "sel " : ""}${why.length ? "attn" : ""}`} title={why.length ? `Needs attention: ${why.join(", ")}` : undefined}>
                <td className="wl-check"><input type="checkbox" aria-label={`Select ${it.ticker}`} checked={sel.includes(it.ticker)} onChange={() => toggle(it.ticker)} /></td>
                <td><TickerLink ticker={it.ticker} named />{why.length > 0 && <span className="sr-only"> Needs attention: {why.join(", ")}</span>}</td>
                {s ? (
                  <>
                    <td className="wl-score"><ScoreBadge score={s.score} />{!it.latest && it.auto && <span className="wl-auto" title="Auto: not ranked in the latest scan, so Premium scored it after the scan">A</span>}{scoreDelta(it.ticker)}</td>
                    <td><span className="wl-setup">{setupLabel(s.primary_setup)}</span>{s.fading && <> <Pill tone="warn">Fading</Pill></>}{x?.prebreakout && <> <Pill tone="gold">PB</Pill></>}</td>
                  </>
                ) : <td colSpan={2} className="cap wl-unranked">{legacy ? "Not among your plan's ranked rows in the latest scan" : "Not ranked in the latest scan"}</td>}
                <td className="num">{priceCell(it)}</td>
                <td className={`num mono ${tone(chgOf(it.ticker))}`}>{pct(chgOf(it.ticker), 1)}</td>
                <td className={`num mono ${tone(since)}`} title={it.added_at ? `Added ${shortDate(it.added_at)}${it.price_when_added ? ` at ${price(it.price_when_added)}` : ""}` : undefined}>{pct(since, 1)}</td>
                <td className="num mono">{s && rank ? <span title={w.scan_total ? `#${rank} of ${w.scan_total} ranked setups` : undefined}>#{rank}<span className="sr-only">{w.scan_total ? ` of ${w.scan_total}` : ""}</span></span> : <span className="cap">—</span>}
                  {s && rank && rc !== 0 && <span className={`wl-delta ${rc > 0 ? "up" : "down"}`} title={`${rc > 0 ? "Up" : "Down"} ${Math.abs(rc)} place${Math.abs(rc) === 1 ? "" : "s"} since the previous scan`}>{rc > 0 ? "▲" : "▼"}{Math.abs(rc)}</span>}</td>
                <td className="num mono wl-mid">{x?.rvol != null ? `${num(x.rvol)}x` : <span className="cap">—</span>}</td>
                <td className="wl-wide"><DailySpark points={series[it.ticker]} ticker={it.ticker} width={78} height={22} /></td>
                <td className="wl-note-cell">{noteBtn(it)}</td>
                <td className="wl-act">{alerts > 0 && <span className="wl-bell" title={`${alerts} active alert${alerts === 1 ? "" : "s"}`} aria-label={`${alerts} alert${alerts === 1 ? "" : "s"}`}>🔔</span>}{removeBtn(it.ticker)}</td>
              </tr>,
              noteOpen === it.ticker && <tr key={`${it.ticker}-note`} className="wl-note-row"><td colSpan={COLS}>{noteEditor(it)}</td></tr>,
            ];
          })}
        </tbody>
      </table>
    </div>
  );

  // Phone: one compact row per ticker; "since added", a score drop and the note go underneath.
  const rows = (
    <ul className="wl-rows">
      {items.map((it) => {
        const s = scores.get(it.ticker);
        const since = sinceOf(it);
        const sc = intelBy.get(it.ticker)?.score_change ?? 0;
        return (
          <li key={it.ticker} className={`wl-row${sel.includes(it.ticker) ? " sel" : ""}`}>
            {showChecks && <input type="checkbox" aria-label={`Select ${it.ticker}`} checked={sel.includes(it.ticker)} onChange={() => toggle(it.ticker)} />}
            <span className="wl-row-score">{s ? <ScoreBadge score={s.score} /> : <span className="score score-weak" title="Not ranked in the latest scan">—</span>}</span>
            <span className="wl-row-main"><TickerLink ticker={it.ticker} logo={false} /> <span className="cap">{s ? setupLabel(s.primary_setup) : "Not ranked"}</span>
              {s && !it.latest && it.auto && <span className="wl-auto" title="Auto-scored after the scan (Premium)">A</span>}{s?.fading && <> <Pill tone="warn">Fading</Pill></>}</span>
            <span className="wl-row-price">{priceCell(it)}</span>
            <span className={`wl-row-chg mono ${tone(chgOf(it.ticker))}`}>{pct(chgOf(it.ticker), 1)}</span>
            <span className="wl-row-sub">
              {since != null && <span className={tone(since)}>{pct(since, 1)} since added</span>}
              {sc < 0 && <span className="down">score −{Math.abs(sc)}</span>}
              {noteBtn(it)}
            </span>
            {noteOpen === it.ticker && <div className="wl-row-note">{noteEditor(it)}</div>}
          </li>
        );
      })}
    </ul>
  );

  // Cards: the roomier layout from before, one card per ticker with every fact spelled out.
  const cards = (
    <ul className="wl-items">
      {items.map((it) => {
        const s = scores.get(it.ticker);
        const bar = daily?.last_bar?.[it.ticker];
        const q = quoteBy[it.ticker];
        const rank = !legacy ? it.latest?.rank : undefined;
        const why = reasonsFor(it.ticker).filter((r) => r !== "Fading" || !s?.fading); // Fading already has its own tag
        return (
          <li key={it.ticker} className="wl-item">
            <div className="wl-main">
              <input type="checkbox" aria-label={`Select ${it.ticker}`} checked={sel.includes(it.ticker)} onChange={() => toggle(it.ticker)} />
              <TickerLink ticker={it.ticker} named />
              {s ? (
                <>
                  <ScoreBadge score={s.score} />
                  <Pill>{setupLabel(s.primary_setup)}</Pill>
                  {!it.latest && it.auto && <AutoTag />}
                  {s.fading && <Pill tone="warn">Fading</Pill>}
                </>
              ) : <span className="cap wl-unranked">{legacy ? "Not among your plan's ranked rows in the latest scan" : "Not ranked in the latest scan"}</span>}
              {why.map((r) => <Pill key={r} tone="warn">{r}</Pill>)}
              <span className="wl-spark"><DailySpark points={series[it.ticker]} ticker={it.ticker} width={88} height={24} /></span>
              <span className="wl-act">
                <Link href={`/stocks/${encodeURIComponent(it.ticker)}`} className="btn btn-sm">Open</Link>
                {removeBtn(it.ticker)}
              </span>
            </div>
            <div className="wl-facts">
              {q ? (
                <>
                  <span className="mono">{price(q.last)}</span>
                  <span className={`mono ${tone(q.chg_pct)}`}>{pct(q.chg_pct)}</span>
                  <span className="cap">{q.source === "daily_bar" ? `daily close${q.at ? ` ${shortDate(q.at)}` : ""} · no recent trade` : q.at ? `last trade ${etTime(q.at)}` : "latest trade"}</span>
                </>
              ) : s ? (
                <>
                  <span className="mono">{price(s.last)}</span>
                  <span className={`mono ${tone(s.chg_pct)}`}>{pct(s.chg_pct)}</span>
                  <span className="cap">at the scan</span>
                </>
              ) : bar ? <span className="mono">{price(bar.close)} <span className="cap">daily bar {shortDate(bar.session)} · may be in progress; not live</span></span> : null}
              {s && rank && <span>#{rank}{w.scan_total ? ` of ${w.scan_total}` : ""}</span>}
              {intelBy.get(it.ticker) && <IntelFacts x={intelBy.get(it.ticker)!} />}
              {sinceOf(it) != null && <span className={tone(sinceOf(it))}>{pct(sinceOf(it), 1)} since added</span>}
              {it.added_at && <span>Added {shortDate(it.added_at)}{it.price_when_added ? ` at ${price(it.price_when_added)}` : ""}</span>}
            </div>
            <NoteEditor wl={w.id} ticker={it.ticker} note={it.note} onSaved={detail.reload} />
          </li>
        );
      })}
    </ul>
  );

  const bulkDone = (message: string) => { setCopying(null); setSelected([]); setBulkMsg(message); reload(); };

  return (
    <div className="stack">
      <Card title={<>{w.name} {w.is_default && <Pill>Default</Pill>}{pinned && <> <Pill>Pinned</Pill></>}{archived && <> <Pill tone="warn">Archived</Pill></>}</>} id="wl-detail"
        aside={`${w.symbol_count} ticker${w.symbol_count === 1 ? "" : "s"}`}>
        {archived && (
          <p className="notice wl-archived-note">This list is archived. It keeps its tickers and notes but sits in the Archived group.{" "}
            <button type="button" className="link-btn" disabled={act.busy}
              onClick={() => void act.run(async () => { await watchlists.setArchived(w.id, false); reload(); return true; })}>Unarchive</button></p>
        )}
        <div className="wl-summary" aria-label="Watchlist summary">
          {chips.map((c) => (
            <button key={c.f} type="button" className={`wl-chip${filter === c.f && c.f !== "all" ? " on" : ""}`}
              aria-pressed={c.f === "all" ? filter === "all" : filter === c.f} title={c.tip}
              aria-label={`${c.n} ${c.label}. ${c.tip}`} disabled={c.f !== "all" && c.n === 0 && filter !== c.f}
              onClick={() => setFilter(c.f)}>
              <strong>{c.n}</strong> {c.label}
            </button>
          ))}
        </div>
        <div className="row-actions watchlist-actions">
          <button type="button" className="btn" aria-expanded={manageOpen || !w.items.length} aria-controls="watchlist-add" onClick={() => setManageOpen((v) => !v)}>Add tickers</button>
          <details className="watchlist-management">
            <summary>Manage list</summary>
            <div className="row-actions wl-manage" aria-label="Watchlist management">
              <button type="button" className="btn" onClick={() => { ren.clear(); setRenaming(true); }}>Rename</button>
              <button type="button" className="btn" disabled={w.is_default || act.busy}
                onClick={() => void act.run(async () => { await watchlists.makeDefault(w.id); reload(); return true; })}>
                {w.is_default ? "Default list" : "Make default"}
              </button>
              {!w.is_default && !archived && (
                <button type="button" className="btn" disabled={act.busy}
                  onClick={() => void act.run(async () => { await watchlists.setPinned(w.id, !pinned); reload(); return true; })}>{pinned ? "Unpin" : "Pin to top"}</button>
              )}
              <button type="button" className="btn" disabled={act.busy || w.is_default} title={w.is_default ? "Your default list can't be archived. Make another list the default first." : undefined}
                onClick={() => void act.run(async () => { await watchlists.setArchived(w.id, !archived); reload(); return true; })}>{archived ? "Unarchive" : "Archive"}</button>
              <button type="button" className="btn" disabled={lists.length < 2} onClick={onMerge}>Merge another list in…</button>
              <button type="button" className="btn btn-danger-outline" onClick={() => setDeleting(true)}>Delete</button>
            </div>
            <ErrorLine error={act.error} />
          </details>
        </div>
        <div id="watchlist-add" hidden={!manageOpen && w.items.length > 0}><AddTickers wl={w.id} onAdded={reload} /></div>
      </Card>

      <div className="wl-tabs" role="tablist" aria-label="Watchlist views" onKeyDown={tabKey}>
        {TABS.map(([k, label]) => (
          <button key={k} type="button" role="tab" id={`wl-tab-${k}`} aria-controls={`wl-panel-${k}`} aria-selected={tab === k}
            tabIndex={tab === k ? 0 : -1} className={`wl-tab${tab === k ? " on" : ""}`} onClick={() => setTab(k)}>
            {label}{k === "changes" && changeCount > 0 && <span className="wl-tab-n">{changeCount}</span>}
          </button>
        ))}
      </div>

      <section id="wl-panel-tickers" role="tabpanel" aria-labelledby="wl-tab-tickers" hidden={tab !== "tickers"}>
      <Card title="Tickers" id="wl-items" aside={scanAt ? <Freshness at={scanAt} label="HSF Scores from the scan" stale={stale} /> : undefined}>
        {w.items.length > 0 && (
          <div className="wl-toolbar">
            {narrow ? (
              <>
                <label className="field wl-sort-select"><span className="sr-only">Sort tickers</span>
                  <select aria-label="Sort tickers" value={`${sort.key}:${sort.desc ? "d" : "a"}`} onChange={(e) => { const [k, d] = e.target.value.split(":"); setPrefs({ ...prefs, sort: { key: k as SortKey, desc: d === "d" } }); }}>
                    {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => <option key={k} value={`${k}:${ASC_FIRST.includes(k) ? "a" : "d"}`}>Sort: {SORT_LABEL[k]}</option>)}
                  </select>
                </label>
                <button type="button" className="btn" aria-pressed={selecting} onClick={() => { setSelecting((v) => !v); setSelected([]); }}>{selecting ? "Done" : "Select"}</button>
              </>
            ) : w.items.length > 1 && (
              <div className="seg-group" role="radiogroup" aria-label="Layout">
                {([["table", "Table"], ["cards", "Cards"]] as const).map(([v, label]) => (
                  <label key={v} className={`seg${prefs.view === v ? " on" : ""}`}>
                    <input type="radio" className="sr-only" name={`wl-view-${w.id}`} checked={prefs.view === v} onChange={() => setPrefs({ ...prefs, view: v })} />{label}
                  </label>
                ))}
              </div>
            )}
            <div className="wl-status">
              <RefreshStatus updatedAt={detail.updatedAt} refreshing={refreshing} error={detail.error} onRetry={refresh}
                sourceNote={priceNote(hasQuotes, quotes.loading && !quotes.data, feed)} />
            </div>
            <button type="button" className="btn wl-refresh" onClick={refresh} disabled={refreshing} aria-label={`Refresh prices and scores for ${w.name}`}>
              {refreshing ? "Refreshing…" : "Refresh"}
            </button>
            <ResearchCsvButton what={`the ${w.name} watchlist`} filename={csvFile("watchlist", w.name)}
              build={() => watchlistCsv(items, scores, quoteBy, scanAt)} />
          </div>
        )}
        {sel.length > 0 && (
          <div className="wl-bulk" role="region" aria-label="Selected tickers">
            <strong>{sel.length} selected</strong>
            <span className="grow" />
            <button type="button" className="btn btn-sm" onClick={() => setCopying("move")}>Move to…</button>
            <button type="button" className="btn btn-sm" onClick={() => setCopying("copy")}>Copy to…</button>
            {sel.length >= 2 && sel.length <= 4
              ? <Link className="btn btn-sm" href={`/stocks/compare?symbols=${encodeURIComponent(sel.join(","))}`}>Compare</Link>
              : <button type="button" className="btn btn-sm" disabled title="Compare takes 2 to 4 stocks">Compare</button>}
            <button type="button" className="btn btn-sm btn-danger-outline" onClick={() => { bulk.clear(); setConfirmBulk(true); }}>Remove selected</button>
            <button type="button" className="link-btn" onClick={() => setSelected([])}>Clear</button>
          </div>
        )}
        {bulkMsg && <p className="notice" role="status">{bulkMsg} <button type="button" className="link-btn" onClick={() => setBulkMsg(null)}>Dismiss</button></p>}
        {filter !== "all" && w.items.length > 0 && (
          <p className="cap wl-filter-note" role="status">
            Showing {items.length} of {w.items.length}: {filterLabel}.{" "}
            <button type="button" className="link-btn" onClick={() => setFilter("all")}>Show all</button>
          </p>
        )}
        {w.items.length === 0 ? (
          <Empty title="No tickers yet.">Add some above, or use Save to watchlist on the Scanner or a stock page.</Empty>
        ) : view === "rows" ? rows : view === "cards" ? cards : table}
        {legacy && scan.data && <p className="cap">Scores and prices are from that scan, only for tickers among your plan&apos;s top {scan.data.max_results} ranked setups. Not live quotes.</p>}
        {!legacy && w.items.some((i) => !i.latest && i.auto) && <p className="cap">{view === "cards" ? "Auto" : "A (Auto)"}: not ranked in the latest scan, so Premium scores it after each scan.{w.auto_scored_at ? ` Scored ${etTime(w.auto_scored_at)}.` : ""}</p>}
        {!legacy && w.items.length > 0 && <p className="cap">{scanAt ? `Score, setup and rank are from that scan. ${hasQuotes ? "Price and change are the latest trade vs the previous close." : "Price and change are from that scan, not live quotes."} The rank is the ticker's place among the scan's ranked setups. Since added compares today's price with the price when you added it.` : "The latest market scan couldn't be read, so scores aren't shown. The list itself is current."}</p>}
      </Card>
      </section>

      <section id="wl-panel-changes" role="tabpanel" aria-labelledby="wl-tab-changes" hidden={tab !== "changes"}>
        <WatchlistChangesCard id={w.id} />
      </section>
      <section id="wl-panel-performance" role="tabpanel" aria-labelledby="wl-tab-performance" hidden={tab !== "performance"}>
        <WatchlistPerformanceCard id={w.id} state={performance} />
      </section>
      <section id="wl-panel-ai" role="tabpanel" aria-labelledby="wl-tab-ai" hidden={tab !== "ai"}>
        <WatchlistAINote id={w.id} name={w.name} empty={w.items.length === 0} />
      </section>
      <section id="wl-panel-alerts" role="tabpanel" aria-labelledby="wl-tab-alerts" hidden={tab !== "alerts"}>
        <WatchlistRules wl={w.id} onChanged={intel.reload} />
      </section>

      <CopyDialog key={`${copying}-${sel.join(",")}`} open={copying !== null} move={copying === "move"} from={w} tickers={sel} lists={lists}
        onClose={() => setCopying(null)} onDone={bulkDone} />
      <ConfirmDialog open={confirmBulk} title={`Remove ${sel.length} ticker${sel.length === 1 ? "" : "s"}?`} confirmLabel="Remove" busy={bulk.busy}
        body={<p>{sel.join(", ")} and their notes will be removed from {w.name}.</p>} error={<ErrorLine error={bulk.error} />}
        onClose={() => setConfirmBulk(false)}
        onConfirm={() => void bulk.run(async () => {
          const r = await watchlists.removeTickers(w.id, sel);
          setConfirmBulk(false);
          bulkDone(`Removed ${r.removed?.length ?? 0} from ${w.name}.`);
          return true;
        })} />
      <NameDialog open={renaming} title="Rename watchlist" initial={w.name} submitLabel="Rename" onClose={() => setRenaming(false)}
        error={<ErrorLine error={ren.error} />}
        onSubmit={async (name) => !!(await ren.run(async () => { await watchlists.rename(w.id, name); reload(); return true; }))} />
      <ConfirmDialog open={deleting} title={`Delete "${w.name}"?`} confirmLabel="Delete watchlist" busy={del.busy}
        body={<p>This removes the list and its {w.symbol_count} ticker{w.symbol_count === 1 ? "" : "s"} and notes. Its list alerts are switched off; ticker alerts aren&apos;t affected. This can&apos;t be undone.{!w.is_default ? " To keep it out of the way instead, use Archive." : ""}</p>}
        error={<ErrorLine error={del.error} />} onClose={() => { del.clear(); setDeleting(false); }}
        onConfirm={() => void del.run(async () => { await watchlists.remove(w.id); setDeleting(false); onDeleted(); return true; })} />
      <ConfirmDialog open={removing !== null} title={`Remove ${removing ?? ""}?`} confirmLabel="Remove" busy={rm.busy}
        body={<p>{removing} and its note will be removed from {w.name}.</p>} error={<ErrorLine error={rm.error} />}
        onClose={() => setRemoving(null)}
        onConfirm={() => void rm.run(async () => { await watchlists.removeTicker(w.id, removing!); setRemoving(null); reload(); return true; })} />
    </div>
  );
}

function ListGroup({ title, lists, selected, onSelect }: { title: string; lists: Watchlist[]; selected: number | undefined; onSelect: (id: number) => void }) {
  if (!lists.length) return null;
  return (
    <div className="wl-group">
      <h3 className="wl-group-h">{title}</h3>
      <ul className="pick-list">
        {lists.map((w) => (
          <li key={w.id}>
            <button type="button" className={`pick${selected === w.id ? " on" : ""}`} aria-current={selected === w.id ? "true" : undefined} onClick={() => onSelect(w.id)}>
              <span className="strong">{w.name}</span>
              {w.is_default && <Pill>Default</Pill>}
              <span className="grow" />
              <span className="cap">{w.symbol_count}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function WatchlistsView() {
  const sp = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const lists = useApi("watchlists", (signal) => watchlists.list(signal));
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState("");
  const [merge, setMerge] = useState<{ target: number; from: number | null } | null>(null);
  const [mergedMsg, setMergedMsg] = useState<string | null>(null);
  const [hideHint, setHideHint] = useState(false);
  const create = useAction();

  const all = lists.data ?? [];
  const asked = Number(sp.get("id"));
  const selected = all.find((w) => w.id === asked) ?? all.find((w) => w.is_default) ?? all.find((w) => !w.archived) ?? all[0] ?? null;
  const select = (id: number | null) => router.replace(id ? `${path}?id=${id}` : path, { scroll: false });
  const groups = groupLists(all, query);
  const archivedCount = all.filter((w) => w.archived).length;
  const dupes = lookAlikes(all);
  const mergeTarget = merge ? all.find((w) => w.id === merge.target) ?? null : null;
  const label = (w: Watchlist) => `${w.name}${w.is_default ? " · Default" : ""} (${w.symbol_count})`;
  const narrowGroups: [string, Watchlist[]][] = [["Pinned", groups.pinned], ["Lists", groups.active], ["Earnings", groups.earnings], ["Archived", groups.archived]];

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Watchlists</h1>
          <p className="cap">Lists of tickers to follow. Notes stay with each ticker.</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => { create.clear(); setCreating(true); }}>New watchlist</button>
      </section>
      {mergedMsg && <p className="notice" role="status">{mergedMsg} <button type="button" className="link-btn" onClick={() => setMergedMsg(null)}>Dismiss</button></p>}

      {lists.error && !lists.data ? <ErrorState error={lists.error} onRetry={lists.reload} what="your watchlists" /> :
        lists.loading && !lists.data ? <Skeleton rows={5} label="Loading watchlists" /> :
        all.length === 0 ? (
          <Card><Empty title="No watchlists yet.">
            <div className="stack-sm">
              <StarterWatchlistButton onDone={(id) => { lists.reload(); select(id); }} />
              <div className="row-actions">
                <button type="button" className="btn" onClick={() => { create.clear(); setCreating(true); }}>Create an empty watchlist</button>
              </div>
            </div>
          </Empty></Card>
        ) : (
          <div className="split watchlists-split">
            <label className="field only-narrow">Current watchlist<select value={selected?.id ?? ""} onChange={(e) => select(Number(e.target.value))}>
              {narrowGroups.map(([g, ls]) => ls.length > 0 && <optgroup key={g} label={g}>{ls.map((w) => <option key={w.id} value={w.id}>{label(w)}</option>)}</optgroup>)}
            </select></label>
            <nav className="col-side card wl-nav" aria-label="Your watchlists">
              {all.length > 6 && (
                <label className="field wl-search"><span className="sr-only">Search watchlists</span>
                  <input type="search" placeholder="Search lists…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search watchlists" />
                </label>
              )}
              <ListGroup title="Pinned" lists={groups.pinned} selected={selected?.id} onSelect={select} />
              <ListGroup title={groups.pinned.length ? "Lists" : "Your lists"} lists={groups.active} selected={selected?.id} onSelect={select} />
              <ListGroup title="Earnings" lists={groups.earnings} selected={selected?.id} onSelect={select} />
              {groups.archived.length > 0 && (
                <details className="wl-archived" open={!!query || !!selected?.archived}>
                  <summary className="wl-group-h">Archived · {groups.archived.length}</summary>
                  <ListGroup title="" lists={groups.archived} selected={selected?.id} onSelect={select} />
                </details>
              )}
              {query && !groups.pinned.length && !groups.active.length && !groups.earnings.length && !groups.archived.length && <p className="cap">No list matches “{query}”.</p>}
              {dupes.length > 0 && !hideHint && (
                <div className="wl-dupes" role="note">
                  <p>{dupes.length === 1 ? "These lists look alike:" : "Some lists look alike:"}</p>
                  <ul>
                    {dupes.map((g) => (
                      <li key={g[0]!.id}>
                        <span>{g.map((w) => w.name).join(" + ")}</span>{" "}
                        <button type="button" className="link-btn" onClick={() => { select(g[0]!.id); setMerge({ target: g[0]!.id, from: g[1]!.id }); }}>Merge</button>
                      </li>
                    ))}
                  </ul>
                  <button type="button" className="link-btn" onClick={() => setHideHint(true)}>Hide</button>
                </div>
              )}
              <p className="cap">{archivedCount ? `${all.length - archivedCount} active · ${archivedCount} archived · ` : ""}{all.length} of 50 watchlists</p>
            </nav>
            <div className="col-main">
              {selected && <Detail key={selected.id} id={selected.id} lists={all} onChanged={lists.reload}
                onDeleted={() => { lists.reload(); select(null); }} onMerge={() => setMerge({ target: selected.id, from: null })} />}
            </div>
          </div>
        )}

      <MergeDialog key={merge ? `${merge.target}-${merge.from}` : "none"} target={mergeTarget} lists={all} initialFrom={merge?.from ?? null}
        onClose={() => setMerge(null)}
        onMerged={(message) => { const t = merge!.target; setMerge(null); setMergedMsg(message); lists.reload(); select(t); }} />
      <NameDialog open={creating} title="New watchlist" withDefault submitLabel="Create" onClose={() => setCreating(false)}
        error={<ErrorLine error={create.error} />}
        onSubmit={async (name, makeDefault) => {
          const wl = await create.run(() => watchlists.create(name, makeDefault));
          if (!wl) return false;
          lists.reload();
          select(wl.id);
          return true;
        }} />
      <p className="cap">Lists sync to your account. <Link href="/alerts">Alerts</Link> can watch these lists.</p>
    </div>
  );
}
