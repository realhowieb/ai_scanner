"use client";

// Trade journal: trades you log yourself (open ones marked to live quotes by the API),
// with closed-trade stats. Logging, closing and deleting are Pro, as in the classic app.
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { TICKER_RE } from "@/components/AppShell";
import { ConfirmDialog, Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Locked, Pill, Skeleton, TickerLink } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime, pct, price } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

type Trade = Schemas["JournalTrade"];
const metric = (v: unknown, fallback = 0) => (typeof v === "number" ? v : fallback);
const tid = (id: number) => ({ params: { path: { trade_id: id } } });
const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

function LogTrade({ onDone }: { onDone: () => void }) {
  const [ticker, setTicker] = useState("");
  const [entry, setEntry] = useState("");
  const [shares, setShares] = useState("");
  const act = useAction();
  const t = ticker.trim().toUpperCase();
  const e = Number(entry);
  const n = Number(shares);
  const valid = TICKER_RE.test(t) && e > 0 && Number.isInteger(n) && n > 0;
  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    if (!valid) return;
    const ok = await act.run(async () => { await unwrap(api.POST("/v1/journal", { body: { ticker: t, entry_price: e, shares: n } })); return true; });
    if (ok) { setTicker(""); setEntry(""); setShares(""); onDone(); }
  };
  return (
    <form className="inline-form" onSubmit={submit} aria-label="Log a trade">
      <label className="field"><span>Ticker</span><input value={ticker} maxLength={10} autoComplete="off" onChange={(x) => setTicker(x.target.value)} /></label>
      <label className="field"><span>Entry price ($)</span><input inputMode="decimal" value={entry} onChange={(x) => setEntry(x.target.value)} /></label>
      <label className="field"><span>Shares</span><input inputMode="numeric" value={shares} onChange={(x) => setShares(x.target.value)} /></label>
      <button type="submit" className="btn btn-primary" disabled={!valid || act.busy}>{act.busy ? "Logging…" : "Log trade"}</button>
      <ErrorLine error={act.error} />
    </form>
  );
}

function Row({ t, pro, onChanged }: { t: Trade; pro: boolean; onChanged: () => void }) {
  const [closing, setClosing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [exit, setExit] = useState("");
  const close = useAction();
  const del = useAction();
  return (
    <tr>
      <td><TickerLink ticker={t.ticker} /> {t.open ? <Pill tone="up">Open</Pill> : <Pill>Closed</Pill>}</td>
      <td className="num mono">{t.shares ?? "—"}</td>
      <td className="num mono">{price(t.entry_price)}</td>
      <td className="num mono">{t.open ? price(t.mark) : price(t.exit_price)}</td>
      <td className={`num mono ${tone(t.pnl)}`}>{t.pnl != null ? `${t.pnl < 0 ? "-" : ""}${price(Math.abs(t.pnl))}` : "—"} <span className="cap">{pct(t.pnl_pct)}</span></td>
      <td className="cap hide-narrow">{etTime(t.open ? t.entered_at : t.closed_at)}</td>
      <td className="num">
        {pro && (
          <div className="row-actions end">
            {t.open && <button type="button" className="btn btn-sm" onClick={() => { close.clear(); setExit(t.mark ? String(Math.round(t.mark * 100) / 100) : ""); setClosing(true); }}>Close</button>}
            <button type="button" className="btn btn-sm btn-danger-outline" aria-label={`Delete the ${t.ticker} trade`} onClick={() => { del.clear(); setDeleting(true); }}>Delete</button>
          </div>
        )}
        <Dialog open={closing} title={`Close ${t.ticker}`} onClose={() => setClosing(false)}>
          <form className="stack-sm" onSubmit={(e) => { e.preventDefault(); if (Number(exit) > 0) void close.run(async () => { await unwrap(api.POST("/v1/journal/{trade_id}/close", { ...tid(t.id), body: { exit_price: Number(exit) } })); setClosing(false); onChanged(); return true; }); }}>
            <label className="field"><span>Exit price ($)</span><input inputMode="decimal" value={exit} onChange={(e) => setExit(e.target.value)} data-autofocus /></label>
            <ErrorLine error={close.error} />
            <div className="row-actions end"><button type="button" className="btn" onClick={() => setClosing(false)}>Cancel</button>
              <button type="submit" className="btn btn-primary" disabled={!(Number(exit) > 0) || close.busy}>{close.busy ? "Closing…" : "Close trade"}</button></div>
          </form>
        </Dialog>
        <ConfirmDialog open={deleting} title={`Delete the ${t.ticker} trade?`} confirmLabel="Delete trade" busy={del.busy}
          body={<p>It disappears from your journal and its stats. This can&apos;t be undone.</p>} error={<ErrorLine error={del.error} />}
          onClose={() => setDeleting(false)}
          onConfirm={() => void del.run(async () => { await unwrap(api.DELETE("/v1/journal/{trade_id}", tid(t.id))); setDeleting(false); onChanged(); return true; })} />
      </td>
    </tr>
  );
}

export function JournalView() {
  const { me } = useSession();
  const pro = me?.plan === "pro" || me?.plan === "premium" || me?.plan === "admin";
  const j = useApi("journal", (signal) => unwrap(api.GET("/v1/journal", { signal })));
  const stats = j.data?.stats as { closed?: number; wins?: number; avg_return_pct?: number | null } | null | undefined;
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Journal</h1>
          <p className="cap">Trades you log yourself. HSF doesn&apos;t place orders. Open trades are marked to live quotes.</p>
        </div>
      </section>
      <Card title="Log a trade" id="log">
        {pro ? <LogTrade onDone={j.reload} /> : <Locked title="Logging trades is part of Pro" plan="pro">Keep a record of your entries and exits with live marks and win rate.</Locked>}
      </Card>
      {j.error && !j.data ? <ErrorState error={j.error} onRetry={j.reload} what="your journal" />
        : !j.data ? <Skeleton rows={5} label="Loading your journal" />
        : j.data.trades.length === 0 ? <Card><Empty title="No trades logged yet.">Log your first trade above, or paper trade from a stock page.</Empty></Card>
        : (
          <Card title="Trades" id="trades" aside={stats?.closed ? `${stats.closed} closed · ${stats.wins ?? 0} wins · avg ${pct(stats.avg_return_pct)}` : undefined}>
            <div className="metric-strip" aria-label="Journal summary">
              <span><strong>{metric(j.data.summary?.open, j.data.trades.filter((t) => t.open).length)}</strong> open</span>
              <span><strong>{metric(j.data.summary?.closed, j.data.trades.filter((t) => !t.open).length)}</strong> closed</span>
              <span><strong>{pct(metric(j.data.summary?.avg_return_pct, stats?.avg_return_pct ?? 0))}</strong> avg return</span>
              <span><strong>{pct(metric(j.data.summary?.win_rate, 0))}</strong> win rate</span>
            </div>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Ticker</th><th scope="col" className="num">Shares</th><th scope="col" className="num">Entry</th>
                  <th scope="col" className="num">Mark / exit</th><th scope="col" className="num">P&amp;L</th><th scope="col" className="hide-narrow">When</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
                <tbody>{j.data.trades.map((t) => <Row key={t.id} t={t} pro={pro} onChanged={j.reload} />)}</tbody>
              </table>
            </div>
            <p className="cap">Your own records, not HSF results. Not financial advice.</p>
          </Card>
        )}
    </div>
  );
}
