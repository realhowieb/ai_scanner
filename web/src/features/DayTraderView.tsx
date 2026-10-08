"use client";

// Day Trader (Pro): live quotes, gap, VWAP, relative volume and the day-trade score for
// a symbol source, from /v1/day-trader. Refreshes every 45 s while a session is trading.
import { useEffect, useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, Empty, ErrorState, Locked, Pill, Skeleton, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { compact, etTime, num, pct, price } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

const SOURCES = [
  { id: "watchlist", label: "My default watchlist" },
  { id: "movers", label: "Top movers" },
  { id: "movers_sp500", label: "S&P 500 movers" },
  { id: "movers_nasdaq", label: "NASDAQ movers" },
  { id: "premarket", label: "Pre-market scan" },
  { id: "postmarket", label: "After-hours scan" },
  { id: "scan_picks", label: "Latest scan picks" },
  { id: "megacaps", label: "Mega caps" },
] as const;
type Source = (typeof SOURCES)[number]["id"];
const STATE: Record<string, string> = { premarket: "Pre-market", open: "Market open", afterhours: "After hours", closed: "Market closed" };
export const DT_REFRESH_MS = 45_000;
const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

function Live() {
  const [source, setSource] = useState<Source>("movers");
  const [tick, setTick] = useState(0);
  const dt = useApi(`dt:${source}:${tick}`, (signal) => unwrap(api.GET("/v1/day-trader", { params: { query: { source } }, signal })));
  const trading = !!dt.data && dt.data.state !== "closed";
  useEffect(() => {
    if (!trading) return;
    const t = setInterval(() => setTick((n) => n + 1), DT_REFRESH_MS);
    return () => clearInterval(t);
  }, [trading]);
  const d = dt.data;
  const rows = d ? [...d.rows].sort((a, b) => b.day_trade_score - a.day_trade_score) : [];
  return (
    <>
      <section className="filters" aria-label="Source">
        <label className="inline-field">Symbols
          <select value={source} onChange={(e) => setSource(e.target.value as Source)}>
            {SOURCES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
        </label>
        <span className="grow" />
        {d && <span className="cap">{STATE[d.state] ?? d.state} · quotes {etTime(d.as_of)}{trading ? " · refreshes every 45s" : ""}</span>}
        <button type="button" className="btn btn-sm" onClick={() => setTick((n) => n + 1)} disabled={dt.loading}>Refresh</button>
      </section>
      {dt.error && !d ? <ErrorState error={dt.error} onRetry={dt.reload} what="Day Trader" />
        : !d ? <Skeleton rows={8} label="Loading live quotes" />
        : rows.length === 0 ? <Card><Empty title={d.symbols.length ? "No live quotes for these symbols right now." : "No symbols in this source."}>
            {source === "watchlist" ? "Add tickers to your default watchlist, or pick another source." : "Try another source."}</Empty></Card>
        : (
          <section className={`card flush${dt.loading ? " dim" : ""}`}>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Ticker</th><th scope="col" className="num">DT score</th><th scope="col" className="num">Last</th><th scope="col" className="num">Chg %</th>
                  <th scope="col" className="num hide-narrow">Gap %</th><th scope="col" className="num hide-narrow">vs VWAP</th><th scope="col" className="num">RVOL</th><th scope="col" className="num hide-narrow">Volume</th></tr></thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.ticker}>
                      <td><TickerLink ticker={r.ticker} /></td>
                      <td className="num mono strong">{Math.round(r.day_trade_score)}</td>
                      <td className="num mono">{price(r.last)}</td>
                      <td className={`num mono ${tone(r.chg_pct)}`}>{pct(r.chg_pct)}</td>
                      <td className="num mono hide-narrow">{pct(r.gap_pct)}</td>
                      <td className={`num mono hide-narrow ${tone(r.vs_vwap_pct)}`}>{pct(r.vs_vwap_pct)}</td>
                      <td className="num mono">{num(r.rvol)}</td>
                      <td className="num mono hide-narrow">{compact(r.volume)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      {d && d.missing > 0 && <p className="cap"><Pill tone="warn">{d.missing} missing</Pill> {d.missing} symbol{d.missing === 1 ? " has" : "s have"} no live quote right now.</p>}
      <p className="cap">DT score is a day-trading momentum ranking from live quotes, separate from the HSF Score. Not financial advice.</p>
    </>
  );
}

export function DayTraderView() {
  const { can } = useSession();
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Day Trader</h1>
          <p className="cap">Live intraday quotes ranked by day-trade score.</p>
        </div>
      </section>
      {can("can_day_trader") ? <Live /> : <Card><Locked title="Day Trader is part of Pro" plan="pro">Live quotes, VWAP and relative volume for movers or your watchlist, refreshed through the session.</Locked></Card>}
    </div>
  );
}
