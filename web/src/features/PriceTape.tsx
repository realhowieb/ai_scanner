"use client";

import { useEffect, useRef, useState } from "react";
import { api, unwrap } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { pct } from "@/lib/format";

type Quote = { symbol: string; last: number; chg_pct?: number | null };

function Items({ quotes, copy }: { quotes: Quote[]; copy: number }) {
  return (
    <>
      {quotes.map((q) => (
        <span key={`${copy}-${q.symbol}`} className="tape-item mono" aria-hidden={copy > 0 || undefined}>
          <span className="strong">{q.symbol}</span> {q.last.toFixed(2)}{" "}
          <span className={q.chg_pct && q.chg_pct > 0 ? "up" : q.chg_pct && q.chg_pct < 0 ? "down" : ""}>{pct(q.chg_pct)}</span>
        </span>
      ))}
    </>
  );
}

/** Seconds one set of quotes takes to scroll past; more copies scroll for longer so the speed stays the same. */
const SECONDS_PER_SET = 40;

/** The scrolling price strip (the Streamlit header's tape). Hidden when there are no quotes;
 * it pauses on hover and holds still for reduced motion. On wide screens one set of quotes is
 * narrower than the strip, so each run repeats the set until it is at least as wide as the strip
 * (otherwise the loop shows an empty gap on the right). */
export function PriceTape() {
  const { data } = useApi("tape", (signal) => unwrap(api.GET("/v1/market/tape", { signal })));
  const quotes = data?.quotes ?? [];
  const stripRef = useRef<HTMLElement>(null);
  const runRef = useRef<HTMLDivElement>(null);
  const [copies, setCopies] = useState(1);

  useEffect(() => {
    const strip = stripRef.current;
    const run = runRef.current;
    if (!strip || !run || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const shown = Number(run.dataset.copies) || 1;
      const setWidth = run.scrollWidth / shown;
      if (setWidth <= 0) return;
      const next = Math.max(1, Math.ceil(strip.clientWidth / setWidth));
      if (next !== shown) setCopies(next);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(strip);
    return () => ro.disconnect();
  }, [quotes.length]);

  if (quotes.length === 0) return null;
  const sets = Array.from({ length: copies }, (_, i) => <Items key={i} quotes={quotes} copy={i} />);
  return (
    <section ref={stripRef} className="tape" aria-label="Market prices">
      <div className="tape-track" style={{ animationDuration: `${SECONDS_PER_SET * copies}s` }}>
        <div ref={runRef} className="tape-run" data-copies={copies}>{sets}</div>
        <div className="tape-run" aria-hidden="true">{sets}</div>
      </div>
    </section>
  );
}
