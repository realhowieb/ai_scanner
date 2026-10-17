"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, ErrorState, Freshness, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { num, pct, price, setupLabel } from "@/lib/format";

export function compareSymbols(raw: string) {
  return [...new Set(raw.toUpperCase().split(/[\s,]+/).filter(Boolean))];
}
const VALID = /^[A-Z][A-Z0-9.\-]{0,9}$/;
function ComparedStock({ ticker }: { ticker: string }) {
  const s = useApi(`stock:${ticker}`, (signal) => unwrap(api.GET("/v1/stocks/{ticker}", { params: { path: { ticker } }, signal })));
  return <Card title={ticker} id={`compare-${ticker}`}>
    {s.error && !s.data ? <ErrorState error={s.error} onRetry={s.reload} what={ticker} /> : !s.data ? <Skeleton rows={8} /> : <>
      <TickerLink ticker={ticker} />
      <Freshness at={s.data.scan_at} />
      {s.data.from_history && <p className="notice">Score is from an earlier scan.</p>}
      {!s.data.has_setup && <p className="cap">No qualifying setup recorded.</p>}
      <dl className="kv">
        <div><dt>HSF Score</dt><dd><ScoreBadge score={s.data.hsf_score} /></dd></div>
        <div><dt>Setup</dt><dd>{setupLabel(s.data.primary_setup)}</dd></div>
        <div><dt>Last</dt><dd>{price(s.data.price)}</dd></div>
        <div><dt>Change</dt><dd>{pct(s.data.change_pct)}</dd></div>
        <div><dt>Relative volume</dt><dd>{num(s.data.rvol)}</dd></div>
        <div><dt>Status</dt><dd>{s.data.status ?? "—"}</dd></div>
        <div><dt>Signals</dt><dd>{s.data.signals.map(setupLabel).join(", ") || "None"}</dd></div>
      </dl>
      <h3 className="strong">Reasons</h3><ul className="bullets">{s.data.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
      <h3 className="strong">Risks</h3><ul className="bullets">{s.data.risks.length ? s.data.risks.map((r) => <li key={r}>{r}</li>) : <li>No risks supplied by this scan.</li>}</ul>
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
  return <div className="stack">
    <section className="page-head"><div><h1 className="h1">Compare stocks</h1><p className="cap">Review 2–4 names together. Each column shows its own scan time.</p></div></section>
    <form className="row-actions" onSubmit={(e) => {
      e.preventDefault(); const list = compareSymbols(input);
      if (list.length < 2 || list.length > 4 || !list.every((s) => VALID.test(s))) { setBad(true); return; }
      setBad(false); router.replace(`/stocks/compare?symbols=${encodeURIComponent(list.join(","))}`);
    }}><label className="field">Tickers<input value={input} onChange={(e) => setInput(e.target.value)} maxLength={100} placeholder="AAPL, MSFT" /></label><button className="btn btn-primary">Compare</button></form>
    {(bad || (raw && !valid)) && <p className="form-error" role="alert">Enter 2–4 different ticker symbols.</p>}
    {valid && <div className="stock-comparison">{symbols.map((ticker) => <ComparedStock key={ticker} ticker={ticker} />)}</div>}
  </div>;
}
