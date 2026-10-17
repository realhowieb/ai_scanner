"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import type { Schemas } from "@/api/client";

import { api, unwrap } from "@/api/client";
import { Card, ErrorState, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { SnapshotTime } from "./SnapshotTime";
import { num, pct, price, setupLabel } from "@/lib/format";

export function compareSymbols(raw: string) {
  return [...new Set(raw.toUpperCase().split(/[\s,]+/).filter(Boolean))];
}
type Stock = Schemas["StockDetail"];
const METRICS = { hsf_score: "HSF Score", primary_setup: "Setup", price: "Last", change_pct: "Change", rvol: "Relative volume", status: "Status" } as const;
type Metric = keyof typeof METRICS;
const VALID = /^[A-Z][A-Z0-9.\-]{0,9}$/;
function ComparedStock({ ticker, differences, report }: { ticker: string; differences: Metric[]; report: (ticker: string, stock: Stock) => void }) {
  const s = useApi(`stock:${ticker}`, (signal) => unwrap(api.GET("/v1/stocks/{ticker}", { params: { path: { ticker } }, signal })));
  useEffect(() => { if (s.data) report(ticker, s.data); }, [s.data, ticker, report]);
  return <Card title={ticker} id={`compare-${ticker}`}>
    {s.error && !s.data ? <ErrorState error={s.error} onRetry={s.reload} what={ticker} /> : !s.data ? <Skeleton rows={8} /> : <>
      <TickerLink ticker={ticker} />
      <SnapshotTime at={s.data.scan_at} archived={s.data.from_history} />
      {s.data.from_history && <p className="notice">Score is from an earlier scan.</p>}
      {!s.data.has_setup && <p className="cap">No qualifying setup recorded.</p>}
      <dl className="kv">
        <div className={differences.includes("hsf_score") ? "compare-difference" : undefined}><dt>HSF Score</dt><dd><ScoreBadge score={s.data.hsf_score} /></dd></div>
        <div className={differences.includes("primary_setup") ? "compare-difference" : undefined}><dt>Setup</dt><dd>{setupLabel(s.data.primary_setup)}</dd></div>
        <div className={differences.includes("price") ? "compare-difference" : undefined}><dt>Last</dt><dd>{price(s.data.price)}</dd></div>
        <div className={differences.includes("change_pct") ? "compare-difference" : undefined}><dt>Change</dt><dd>{pct(s.data.change_pct)}</dd></div>
        <div className={differences.includes("rvol") ? "compare-difference" : undefined}><dt>Relative volume</dt><dd>{num(s.data.rvol)}</dd></div>
        <div className={differences.includes("status") ? "compare-difference" : undefined}><dt>Status</dt><dd>{s.data.status ?? "—"}</dd></div>
        <div><dt>Signals</dt><dd>{s.data.signals.map(setupLabel).join(", ") || "None"}</dd></div>
      </dl>
      <p className="cap">— means this scan did not return a value.</p>
      <h3 className="strong">Reasons</h3><ul className="bullets">{s.data.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
      <h3 className="strong">Risks</h3><ul className="bullets">{s.data.risks.length ? s.data.risks.map((r) => <li key={r}>{r}</li>) : <li>No risk details returned by this scan. This is not an all-clear.</li>}</ul>
      <h3 className="strong">Watch next</h3><ul className="bullets">{s.data.watch_next.map((r) => <li key={r}>{r}</li>)}</ul>
    </>}
  </Card>;
}
export function CompareStocksView() {
  const params = useSearchParams();
  const router = useRouter();
  const raw = params.get("symbols") ?? "";
  const symbols = compareSymbols(raw);
  const valid = symbols.length >= 2 && symbols.length <= 4 && symbols.every((s) => VALID.test(s));
  const [input, setInput] = useState(raw);
  const [bad, setBad] = useState(false);
  const [loaded, setLoaded] = useState<Record<string, Stock>>({});
  const report = useCallback((ticker: string, stock: Stock) => setLoaded((old) => old[ticker] === stock ? old : { ...old, [ticker]: stock }), []);
  const complete = symbols.every((ticker) => loaded[ticker]);
  const differences = complete ? (Object.keys(METRICS) as Metric[]).filter((key) => new Set(symbols.map((ticker) => loaded[ticker]?.[key] ?? null)).size > 1) : [];
  return <div className="stack">
    <section className="page-head"><div><h1 className="h1">Compare stocks</h1><p className="cap">Review 2–4 names together. Each column shows its own scan time.</p></div></section>
    <form className="row-actions" onSubmit={(e) => {
      e.preventDefault(); const list = compareSymbols(input);
      if (list.length < 2 || list.length > 4 || !list.every((s) => VALID.test(s))) { setBad(true); return; }
      setBad(false); router.replace(`/stocks/compare?symbols=${encodeURIComponent(list.join(","))}`);
    }}><label className="field">Tickers<input value={input} onChange={(e) => setInput(e.target.value)} maxLength={100} placeholder="AAPL, MSFT" /></label><button className="btn btn-primary">Compare</button></form>
    {(bad || (raw && !valid)) && <p className="form-error" role="alert">Enter 2–4 different ticker symbols.</p>}
    {valid && <><p className="cap">{complete ? differences.length ? `Highlighted differences: ${differences.map((key) => METRICS[key]).join(", ")}.` : "These summary metrics match across the selected stocks." : "Differences appear once all selected stocks have loaded."}</p><div className="stock-comparison">{symbols.map((ticker) => <ComparedStock key={ticker} ticker={ticker} differences={differences} report={report} />)}</div></>}
  </div>;
}
