"use client";

// How my stocks did: each ticker's move since it was added to this watchlist, next to SPY
// over the same span. The API picks the starting price (the price saved when added, else
// the last daily close at that moment) and the latest daily close; nothing is estimated
// here. Hidden when the API has no such route.
import { useState } from "react";

import { watchlists } from "@/api/userData";
import { Card, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { pct, price, shortDate } from "@/lib/format";

const SHOWN = 8;
const tone = (v: number | null | undefined) => ((v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "cap");

export function WatchlistPerformanceCard({ id, version = 0 }: { id: number; version?: number }) {
  const perf = useApi(`wl-perf:${id}`, (signal) => watchlists.performance(id, signal), version);
  const [all, setAll] = useState(false);
  const d = perf.data;
  if (perf.error || !d || !Array.isArray(d.items) || !d.items.length) return null;
  const rows = all ? d.items : d.items.slice(0, SHOWN);
  const s = d.summary;
  return (
    <Card title="How your stocks did" id="wl-perf" aside={d.as_of ? `Daily closes through ${shortDate(d.as_of)}` : undefined}>
      {s ? (
        <div className="metric-strip" aria-label="Watchlist performance summary">
          <span><strong className={`mono ${tone(s.avg_change_pct)}`}>{pct(s.avg_change_pct)}</strong> average</span>
          <span><strong>{s.up}</strong> up · <strong>{s.down}</strong> down</span>
          <span><strong>{s.beat_benchmark}</strong> of {s.compared} beat {d.benchmark}</span>
          <span>Best <strong className="mono">{s.best.ticker}</strong> {pct(s.best.change_pct)}</span>
        </div>
      ) : <p className="cap">No prices yet for these tickers.</p>}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr><th scope="col">Ticker</th><th scope="col" className="hide-narrow">Since</th><th scope="col" className="num hide-narrow">From</th>
              <th scope="col" className="num">Now</th><th scope="col" className="num">Change</th><th scope="col" className="num hide-narrow">vs {d.benchmark}</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.ticker}>
                <td><TickerLink ticker={r.ticker} /></td>
                <td className="hide-narrow cap">{r.base_date ? `${shortDate(r.base_date)}${r.since_first_bar ? " (oldest bar)" : ""}` : "—"}</td>
                <td className="num mono hide-narrow">{price(r.base_price)}</td>
                <td className="num mono">{price(r.last)}</td>
                <td className={`num mono ${tone(r.change_pct)}`}>{pct(r.change_pct)}</td>
                <td className={`num mono hide-narrow ${tone(r.vs_benchmark)}`}>{r.vs_benchmark == null ? "—" : `${r.vs_benchmark > 0 ? "+" : ""}${r.vs_benchmark.toFixed(2)} pts`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {d.items.length > SHOWN && (
        <button type="button" className="link-btn cap" aria-expanded={all} onClick={() => setAll((v) => !v)}>
          {all ? "Show fewer" : `Show all ${d.items.length}`}
        </button>
      )}
      <p className="cap">From the price when you added each stock (or the last close before that) to the latest daily close. Past moves don&apos;t predict future ones.</p>
    </Card>
  );
}
