"use client";

import { api, unwrap } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { pct } from "@/lib/format";

type Quote = { symbol: string; last: number; chg_pct?: number | null };

function Items({ quotes }: { quotes: Quote[] }) {
  return (
    <>
      {quotes.map((q) => (
        <span key={q.symbol} className="tape-item mono">
          <span className="strong">{q.symbol}</span> {q.last.toFixed(2)}{" "}
          <span className={q.chg_pct && q.chg_pct > 0 ? "up" : q.chg_pct && q.chg_pct < 0 ? "down" : ""}>{pct(q.chg_pct)}</span>
        </span>
      ))}
    </>
  );
}

/** The scrolling price strip (the Streamlit header's tape). Hidden when there are no quotes;
 * it pauses on hover and holds still for reduced motion. */
export function PriceTape() {
  const { data } = useApi("tape", (signal) => unwrap(api.GET("/v1/market/tape", { signal })));
  const quotes = data?.quotes ?? [];
  if (quotes.length === 0) return null;
  return (
    <section className="tape" aria-label="Market prices">
      <div className="tape-track">
        <div className="tape-run"><Items quotes={quotes} /></div>
        <div className="tape-run" aria-hidden="true"><Items quotes={quotes} /></div>
      </div>
    </section>
  );
}
