"use client";

// Paper trading (Premium): connect an Alpaca PAPER account (keys are validated by the
// API, stored encrypted and never sent back), place whole-share market buys after an
// explicit confirmation, and see positions and the order feed. No real money moves;
// orders are imported into the Journal.
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { TICKER_RE } from "@/components/AppShell";
import { Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Locked, Pill, Skeleton, TickerLink } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime, pct, price } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

type Status = Schemas["PaperStatus"];
type Position = { symbol?: string; qty?: string | number; avg_entry_price?: string | number; current_price?: string | number;
  market_value?: string | number; unrealized_pl?: string | number; unrealized_plpc?: string | number };
type Order = { order_id?: string; symbol?: string; side?: string; qty?: string | number; filled_qty?: string | number;
  status?: string; filled_avg_price?: string | number | null; submitted_at?: string | null; filled_at?: string | null };

/** Alpaca sends numbers as strings. */
export const n = (v: unknown): number | null => {
  const x = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(x) ? x : null;
};
const money = (v: unknown) => {
  const x = n(v);
  return x === null ? "—" : x.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
};
const metric = (v: unknown, fallback = 0) => (typeof v === "number" || typeof v === "string" ? v : fallback);
const tone = (v: number | null) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

function Connect({ onDone }: { onDone: (s: Status) => void }) {
  const [key, setKey] = useState("");
  const [secret, setSecret] = useState("");
  const act = useAction();
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const ok = await act.run(async () => {
      onDone(await unwrap(api.POST("/v1/paper/account", { body: { api_key: key.trim(), api_secret: secret.trim() } })));
      return true;
    });
    if (ok) {
      setKey("");
      setSecret("");
    }
  };
  return (
    <Card title="Connect your Alpaca paper account" id="connect">
      <p className="body-sm">Create a free paper-trading account at Alpaca, then paste its <b>paper</b> API key ID and secret. HSF checks them against Alpaca&apos;s paper endpoint before saving; live-trading keys are refused.</p>
      <form className="stack-sm" onSubmit={submit} autoComplete="off">
        <label className="field"><span>API key ID</span>
          <input value={key} onChange={(e) => setKey(e.target.value)} spellCheck={false} autoComplete="off" required minLength={8} maxLength={128} />
        </label>
        <label className="field"><span>API secret</span>
          <input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} autoComplete="new-password" required minLength={8} maxLength={256} />
        </label>
        <ErrorLine error={act.error} />
        <div className="row-actions">
          <button type="submit" className="btn btn-primary" disabled={act.busy || key.trim().length < 8 || secret.trim().length < 8}>
            {act.busy ? "Checking keys…" : "Connect paper account"}
          </button>
        </div>
      </form>
      <p className="cap">The secret is stored encrypted and never shown again. You can disconnect at any time.</p>
    </Card>
  );
}

function AccountCard({ s, onDisconnected }: { s: Status; onDisconnected: () => void }) {
  const [open, setOpen] = useState(false);
  const act = useAction();
  const a = (s.account ?? null) as { status?: string; buying_power?: unknown; cash?: unknown } | null;
  const disconnect = async () => {
    const ok = await act.run(async () => { await unwrap(api.DELETE("/v1/paper/account")); return true; });
    if (ok) {
      setOpen(false);
      onDisconnected();
    }
  };
  return (
    <Card title="Paper account" id="account" aside={<Pill tone="up">Connected</Pill>}>
      {a ? (
        <dl className="kv">
          <div><dt>Status</dt><dd>{a.status ?? "—"}</dd></div>
          <div><dt>Buying power</dt><dd className="mono">{money(a.buying_power)}</dd></div>
          <div><dt>Cash</dt><dd className="mono">{money(a.cash)}</dd></div>
        </dl>
      ) : <p className="cap">Connected, but Alpaca didn&apos;t answer just now. Balances will show when it does.</p>}
      {s.connected_at && <p className="cap">Connected {etTime(s.connected_at)}.</p>}
      <div className="row-actions"><button type="button" className="btn btn-danger-outline" onClick={() => setOpen(true)}>Disconnect…</button></div>
      <Dialog open={open} title="Disconnect your paper account?" onClose={() => { setOpen(false); act.clear(); }}>
        <div className="stack-sm">
          <p className="body-sm">HSF deletes the stored keys. Your Alpaca paper account and its positions stay as they are, and past orders stay in your Journal.</p>
          <ErrorLine error={act.error} />
          <div className="row-actions end">
            <button type="button" className="btn" onClick={() => setOpen(false)}>Keep connected</button>
            <button type="button" className="btn btn-danger" disabled={act.busy} onClick={() => void disconnect()}>{act.busy ? "Disconnecting…" : "Disconnect"}</button>
          </div>
        </div>
      </Dialog>
    </Card>
  );
}

function OrderCard({ initial, onPlaced }: { initial: string; onPlaced: () => void }) {
  const [ticker, setTicker] = useState(initial);
  const [qty, setQty] = useState("1");
  const [review, setReview] = useState(false);
  const [placed, setPlaced] = useState<Schemas["PaperOrder"] | null>(null);
  const act = useAction();
  const t = ticker.trim().toUpperCase();
  const q = Number(qty);
  const valid = TICKER_RE.test(t) && Number.isInteger(q) && q >= 1 && q <= 100_000;
  const quote = useApi(review && valid ? `paper-quote:${t}` : null, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}", { params: { path: { ticker: t } }, signal })));
  const last = quote.data?.price ?? null;
  const send = async () => {
    const ok = await act.run(async () => {
      setPlaced(await unwrap(api.POST("/v1/paper/orders", { body: { ticker: t, qty: q, confirm: true } })));
      return true;
    });
    if (ok) {
      setReview(false);
      onPlaced();
    }
  };
  return (
    <Card title="Paper trade a setup" id="order">
      <form className="inline-form" onSubmit={(e) => { e.preventDefault(); if (valid) { setPlaced(null); act.clear(); setReview(true); } }}>
        <label className="field"><span>Ticker</span>
          <input value={ticker} maxLength={10} autoComplete="off" onChange={(e) => setTicker(e.target.value)} aria-invalid={(!!ticker && !TICKER_RE.test(t)) || undefined} />
        </label>
        <label className="field"><span>Shares</span>
          <input type="number" inputMode="numeric" min={1} max={100000} step={1} value={qty} onChange={(e) => setQty(e.target.value)} />
        </label>
        <button type="submit" className="btn btn-primary" disabled={!valid}>Review order</button>
      </form>
      <p className="cap">Whole-share market buy in your Alpaca paper account. No real money. The order is added to your <Link href="/journal">Journal</Link> with the trade plan&apos;s stop and target.</p>
      {placed && <p className="notice" role="status">Sent: buy {placed.qty} {placed.ticker} ({placed.status}{n(placed.filled_avg_price) !== null ? `, filled at ${price(n(placed.filled_avg_price))}` : ""}).</p>}
      <Dialog open={review} title="Confirm paper order" onClose={() => { setReview(false); act.clear(); }}>
        <div className="stack-sm">
          <dl className="kv">
            <div><dt>Order</dt><dd>Market buy</dd></div>
            <div><dt>Ticker</dt><dd className="mono">{t}</dd></div>
            <div><dt>Shares</dt><dd className="mono">{q}</dd></div>
            <div><dt>Last price</dt><dd className="mono">{quote.loading ? "…" : last !== null ? price(last) : "—"}</dd></div>
            {last !== null && <div><dt>About</dt><dd className="mono">{money(last * q)}</dd></div>}
          </dl>
          <p className="cap">Paper account only. Market orders fill at the next available price, which can differ from the last price.</p>
          <ErrorLine error={act.error} />
          <div className="row-actions end">
            <button type="button" className="btn" onClick={() => setReview(false)}>Cancel</button>
            <button type="button" className="btn btn-primary" disabled={act.busy} onClick={() => void send()}>{act.busy ? "Sending…" : `Buy ${q} ${t}`}</button>
          </div>
        </div>
      </Dialog>
    </Card>
  );
}

function Activity({ tick }: { tick: number }) {
  const act = useApi(`paper-activity:${tick}`, (signal) => unwrap(api.GET("/v1/paper/activity", { params: { query: { limit: 25 } }, signal })));
  if (act.error && !act.data) return <ErrorState error={act.error} onRetry={act.reload} what="your paper positions" />;
  if (!act.data) return <Skeleton rows={6} label="Loading positions and orders" />;
  const positions = act.data.positions as Position[];
  const orders = act.data.orders as Order[];
  return (
    <>
      <Card title="Paper activity" id="activity-summary">
        <div className="metric-strip" aria-label="Paper activity summary">
          <span><strong>{metric(act.data.summary?.positions, positions.length)}</strong> positions</span>
          <span><strong>{metric(act.data.summary?.orders, orders.length)}</strong> orders</span>
          <span><strong>{money(act.data.summary?.market_value)}</strong> value</span>
          <span><strong>{money(act.data.summary?.unrealized_pl)}</strong> open P/L</span>
        </div>
      </Card>
      <Card title="Positions" id="positions">
        {act.data.positions_available === false ? <p className="cap">Alpaca didn&apos;t return positions just now. <button type="button" className="link-btn" onClick={act.reload}>Try again</button></p>
          : positions.length === 0 ? <Empty title="No open paper positions." />
          : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Ticker</th><th scope="col" className="num">Shares</th><th scope="col" className="num hide-narrow">Avg entry</th>
                  <th scope="col" className="num">Last</th><th scope="col" className="num hide-narrow">Value</th><th scope="col" className="num">P/L</th></tr></thead>
                <tbody>
                  {positions.map((p) => {
                    const pl = n(p.unrealized_pl);
                    const plpc = n(p.unrealized_plpc);
                    return (
                      <tr key={p.symbol}>
                        <td>{p.symbol ? <TickerLink ticker={p.symbol} /> : "—"}</td>
                        <td className="num mono">{n(p.qty) ?? "—"}</td>
                        <td className="num mono hide-narrow">{price(n(p.avg_entry_price))}</td>
                        <td className="num mono">{price(n(p.current_price))}</td>
                        <td className="num mono hide-narrow">{money(p.market_value)}</td>
                        <td className={`num mono ${tone(pl)}`}>{money(pl)}{plpc !== null ? ` (${pct(plpc * 100)})` : ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
      </Card>
      <Card title="Orders" id="orders">
        {orders.length === 0 ? <Empty title="No paper orders yet." /> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th scope="col">Ticker</th><th scope="col">Side</th><th scope="col" className="num">Shares</th>
                <th scope="col">Status</th><th scope="col" className="num hide-narrow">Fill</th><th scope="col" className="hide-narrow">Sent</th></tr></thead>
              <tbody>
                {orders.map((o, i) => (
                  <tr key={o.order_id ?? i}>
                    <td>{o.symbol ? <TickerLink ticker={o.symbol} /> : "—"}</td>
                    <td>{o.side ?? "—"}</td>
                    <td className="num mono">{n(o.filled_qty) ? `${n(o.filled_qty)}/${n(o.qty) ?? "?"}` : n(o.qty) ?? "—"}</td>
                    <td>{o.status ?? "—"}</td>
                    <td className="num mono hide-narrow">{price(n(o.filled_avg_price))}</td>
                    <td className="cap hide-narrow">{etTime(o.submitted_at ?? null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}

function Paper() {
  const initial = (useSearchParams().get("ticker") || "").toUpperCase();
  const status = useApi("paper-status", (signal) => unwrap(api.GET("/v1/paper/account", { signal })));
  const [override, setOverride] = useState<Status | null>(null);
  const [tick, setTick] = useState(0);
  const s = override ?? status.data;
  if (status.error && !s) {
    if (status.error.status === 403) return <Card><Locked title="Paper trading is part of Premium" plan="premium">Practice the setups HSF finds in an Alpaca paper account.</Locked></Card>;
    if (status.error.status === 503) return <Card><Empty title="Paper trading isn't available right now.">The server can&apos;t reach the paper-trading service. Try again later.</Empty></Card>;
    return <ErrorState error={status.error} onRetry={status.reload} what="your paper account" />;
  }
  if (!s) return <Skeleton rows={6} label="Loading your paper account" />;
  if (!s.connected) return <Connect onDone={(x) => { setOverride(x); setTick((t) => t + 1); }} />;
  return (
    <div className="split">
      <div className="col-main">
        <OrderCard initial={TICKER_RE.test(initial) ? initial : ""} onPlaced={() => setTick((t) => t + 1)} />
        <Activity tick={tick} />
      </div>
      <aside className="col-side">
        <AccountCard s={s} onDisconnected={() => setOverride({ connected: false })} />
      </aside>
    </div>
  );
}

export function PaperView() {
  const { can } = useSession();
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Paper trading</h1>
          <p className="cap">Practice HSF setups with simulated money in an Alpaca paper account.</p>
        </div>
      </section>
      {can("can_paper_trade") ? <Paper /> : <Card><Locked title="Paper trading is part of Premium" plan="premium">Connect an Alpaca paper account and paper trade setups, with orders added to your Journal.</Locked></Card>}
    </div>
  );
}
