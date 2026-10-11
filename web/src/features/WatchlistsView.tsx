"use client";

// Watchlists: list, create, rename, make default, delete; tickers with notes. Every
// change is shown only after the server confirms it (then the affected data reloads).
// Scores and prices come from the latest market scan, labelled with its time: the API
// attaches each ticker's scan row (items[].latest); an older API without it falls back to
// matching the plan's ranked rows. Tickers that aren't ranked setups show no score
// rather than a made-up quote. GET /v1/watchlists/{id}/intelligence adds what changed since
// the previous scan (score, rank), PreBreakout (Premium), RVOL and alert counts; all of it
// is computed by the API, and an API without that route just leaves those facts out.
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import { alertRules, parseTickers, watchlists } from "@/api/userData";
import type { WatchlistDetail, WatchlistIntelItem } from "@/api/userData";
import { ConfirmDialog, Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Freshness, Pill, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { num, pct, price, setupLabel, shortDate } from "@/lib/format";

const NOTE_MAX = 500;

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

function NoteEditor({ wl, ticker, note, onSaved }: { wl: number; ticker: string; note: string | null | undefined; onSaved: () => void }) {
  const [editing, setEditing] = useState(false);
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
      {sc !== 0 && <span className={`mono ${sc > 0 ? "up" : "down"}`} title="HSF Score change since the previous scan">{sc > 0 ? "▲" : "▼"}{Math.abs(sc)}</span>}
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
  if (rules.error || types.error || !rules.data || !types.data) return null; // an API without rules: no card
  const available = types.data.filter((t) => t.available);
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

function Detail({ id, onChanged, onDeleted }: { id: number; onChanged: () => void; onDeleted: () => void }) {
  const detail = useApi(`wl:${id}`, (signal) => watchlists.get(id, signal));
  const intel = useApi(`wl-intel:${id}`, (signal) => watchlists.intelligence(id, signal));
  const intelBy = useMemo(() => new Map((intel.data?.items ?? []).map((i) => [i.ticker, i])), [intel.data]);
  // Only an API that predates items[].latest (no scan_at key at all) needs the ranked rows.
  const legacy = !!detail.data && detail.data.scan_at === undefined;
  const scan = useApi(legacy ? "wl-scan" : null, (signal) => unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: 200 } }, signal })));
  const [byScore, setByScore] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const act = useAction();
  const ren = useAction();
  const del = useAction();
  const rm = useAction();
  const reload = () => { detail.reload(); intel.reload(); onChanged(); };

  const scores = useMemo(() => {
    if (!legacy) return new Map((detail.data?.items ?? []).filter((i) => i.latest).map((i) => [i.ticker, i.latest!]));
    return new Map((scan.data?.setups ?? []).map((s) => [s.ticker, s]));
  }, [legacy, detail.data, scan.data]);

  if (detail.error && !detail.data) return <ErrorState error={detail.error} onRetry={detail.reload} what="this watchlist" />;
  if (!detail.data || detail.loading && detail.data.id !== id) return <Skeleton rows={5} label="Loading watchlist" />;
  const w: WatchlistDetail = detail.data;
  const scanAt = legacy ? scan.data?.scan_at : w.scan_at;
  const stale = legacy ? scan.data?.stale : w.stale;
  const items = byScore ? [...w.items].sort((a, b) => (scores.get(b.ticker)?.score ?? -1) - (scores.get(a.ticker)?.score ?? -1)) : w.items;
  const ranked = w.items.filter((i) => scores.has(i.ticker)).length;
  const attention = w.items.filter((i) => scores.get(i.ticker)?.fading || (intelBy.get(i.ticker)?.score_change ?? 0) < 0).length;
  const notes = w.items.filter((i) => i.note).length;

  return (
    <div className="stack">
      <Card title={<>{w.name} {w.is_default && <Pill>Default</Pill>}</>} id="wl-detail"
        aside={`${w.symbol_count} ticker${w.symbol_count === 1 ? "" : "s"}`}>
        <div className="wl-summary" aria-label="Watchlist summary">
          <span><strong>{w.symbol_count}</strong> tracked</span>
          <span><strong>{ranked}</strong> ranked</span>
          <span><strong>{attention}</strong> need attention</span>
          <span><strong>{notes}</strong> with notes</span>
        </div>
        <div className="row-actions">
          <button type="button" className="btn" onClick={() => { ren.clear(); setRenaming(true); }}>Rename</button>
          <button type="button" className="btn" disabled={w.is_default || act.busy}
            onClick={() => void act.run(async () => { await watchlists.makeDefault(w.id); reload(); return true; })}>
            {w.is_default ? "Default list" : "Make default"}
          </button>
          <button type="button" className="btn btn-danger-outline" onClick={() => setDeleting(true)}>Delete</button>
        </div>
        <ErrorLine error={act.error} />
        <AddTickers wl={w.id} onAdded={reload} />
      </Card>

      <Card title="Tickers" id="wl-items" aside={scanAt ? <Freshness at={scanAt} label="HSF Scores from the scan" stale={stale} /> : undefined}>
        {w.items.length > 1 && (
          <div className="seg-group sort-toggle" role="radiogroup" aria-label="Sort tickers">
            {([[false, "A to Z"], [true, "HSF Score"]] as const).map(([v, label]) => (
              <label key={label} className={`seg${byScore === v ? " on" : ""}`}>
                <input type="radio" className="sr-only" name={`wl-sort-${w.id}`} checked={byScore === v} onChange={() => setByScore(v)} />{label}
              </label>
            ))}
          </div>
        )}
        {w.items.length === 0 ? (
          <Empty title="No tickers yet.">Add some above, or use Save to watchlist on the Scanner or a stock page.</Empty>
        ) : (
          <ul className="wl-items">
            {items.map((it) => {
              const s = scores.get(it.ticker);
              const rank = !legacy ? it.latest?.rank : undefined;
              return (
                <li key={it.ticker} className="wl-item">
                  <div className="wl-main">
                    <TickerLink ticker={it.ticker} />
                    {s ? (
                      <>
                        <ScoreBadge score={s.score} />
                        <Pill>{setupLabel(s.primary_setup)}</Pill>
                        {s.fading && <Pill tone="warn">Fading</Pill>}
                      </>
                    ) : <span className="cap">{legacy ? "Not among your plan's ranked rows in the latest scan" : "Not ranked in the latest scan"}</span>}
                    <span className="grow" />
                    <Link href={`/stocks/${encodeURIComponent(it.ticker)}`} className="btn btn-sm">Open</Link>
                    <button type="button" className="icon-btn" aria-label={`Remove ${it.ticker} from ${w.name}`} onClick={() => { rm.clear(); setRemoving(it.ticker); }}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
                    </button>
                  </div>
                  <div className="wl-facts">
                    {s && (
                      <>
                        <span className="mono">{price(s.last)}</span>
                        <span className={`mono ${(s.chg_pct ?? 0) > 0 ? "up" : (s.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(s.chg_pct)}</span>
                        {rank && <span>#{rank}{w.scan_total ? ` of ${w.scan_total}` : ""}</span>}
                      </>
                    )}
                    {intelBy.get(it.ticker) && <IntelFacts x={intelBy.get(it.ticker)!} />}
                    {it.added_at && <span>Added {shortDate(it.added_at)}{it.price_when_added ? ` at ${price(it.price_when_added)}` : ""}</span>}
                  </div>
                  <NoteEditor wl={w.id} ticker={it.ticker} note={it.note} onSaved={detail.reload} />
                </li>
              );
            })}
          </ul>
        )}
        {legacy && scan.data && <p className="cap">Scores and prices are from that scan, only for tickers among your plan&apos;s top {scan.data.max_results} ranked setups. Not live quotes.</p>}
        {!legacy && w.items.length > 0 && <p className="cap">{scanAt ? "Score, setup, price and change are from that scan, not live quotes. The rank is the ticker's place among the scan's ranked setups." : "The latest market scan couldn't be read, so scores aren't shown. The list itself is current."}</p>}
      </Card>

      <WatchlistRules wl={w.id} onChanged={intel.reload} />

      <NameDialog open={renaming} title="Rename watchlist" initial={w.name} submitLabel="Rename" onClose={() => setRenaming(false)}
        error={<ErrorLine error={ren.error} />}
        onSubmit={async (name) => !!(await ren.run(async () => { await watchlists.rename(w.id, name); reload(); return true; }))} />
      <ConfirmDialog open={deleting} title={`Delete "${w.name}"?`} confirmLabel="Delete watchlist" busy={del.busy}
        body={<p>This removes the list and its {w.symbol_count} ticker{w.symbol_count === 1 ? "" : "s"} and notes. Its list alerts are switched off; ticker alerts aren&apos;t affected. This can&apos;t be undone.</p>}
        error={<ErrorLine error={del.error} />} onClose={() => { del.clear(); setDeleting(false); }}
        onConfirm={() => void del.run(async () => { await watchlists.remove(w.id); setDeleting(false); onDeleted(); return true; })} />
      <ConfirmDialog open={removing !== null} title={`Remove ${removing ?? ""}?`} confirmLabel="Remove" busy={rm.busy}
        body={<p>{removing} and its note will be removed from {w.name}.</p>} error={<ErrorLine error={rm.error} />}
        onClose={() => setRemoving(null)}
        onConfirm={() => void rm.run(async () => { await watchlists.removeTicker(w.id, removing!); setRemoving(null); reload(); return true; })} />
    </div>
  );
}

export function WatchlistsView() {
  const sp = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const lists = useApi("watchlists", (signal) => watchlists.list(signal));
  const [creating, setCreating] = useState(false);
  const create = useAction();

  const all = lists.data ?? [];
  const asked = Number(sp.get("id"));
  const selected = all.find((w) => w.id === asked) ?? all.find((w) => w.is_default) ?? all[0] ?? null;
  const select = (id: number | null) => router.replace(id ? `${path}?id=${id}` : path, { scroll: false });

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Watchlists</h1>
          <p className="cap">Lists of tickers to follow. Notes stay with each ticker.</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => { create.clear(); setCreating(true); }}>New watchlist</button>
      </section>

      {lists.error && !lists.data ? <ErrorState error={lists.error} onRetry={lists.reload} what="your watchlists" /> :
        lists.loading && !lists.data ? <Skeleton rows={5} label="Loading watchlists" /> :
        all.length === 0 ? (
          <Card><Empty title="No watchlists yet.">
            <button type="button" className="btn btn-primary" onClick={() => { create.clear(); setCreating(true); }}>Create your first watchlist</button>
          </Empty></Card>
        ) : (
          <div className="split">
            <nav className="col-side card wl-nav" aria-label="Your watchlists">
              <ul className="pick-list">
                {all.map((w) => (
                  <li key={w.id}>
                    <button type="button" className={`pick${selected?.id === w.id ? " on" : ""}`} aria-current={selected?.id === w.id ? "true" : undefined} onClick={() => select(w.id)}>
                      <span className="strong">{w.name}</span>
                      {w.is_default && <Pill>Default</Pill>}
                      <span className="grow" />
                      <span className="cap">{w.symbol_count}</span>
                    </button>
                  </li>
                ))}
              </ul>
              <p className="cap">{all.length} of 50 watchlists</p>
            </nav>
            <div className="col-main">
              {selected && <Detail key={selected.id} id={selected.id} onChanged={lists.reload}
                onDeleted={() => { lists.reload(); select(null); }} />}
            </div>
          </div>
        )}

      <NameDialog open={creating} title="New watchlist" withDefault submitLabel="Create" onClose={() => setCreating(false)}
        error={<ErrorLine error={create.error} />}
        onSubmit={async (name, makeDefault) => {
          const wl = await create.run(() => watchlists.create(name, makeDefault));
          if (!wl) return false;
          lists.reload();
          select(wl.id);
          return true;
        }} />
      <p className="cap">Changes here also show in the classic app (it may take up to 2 minutes there). <Link href="/alerts">Alerts</Link> can watch these lists.</p>
    </div>
  );
}
