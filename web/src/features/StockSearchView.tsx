"use client";

// Stock Intelligence home: open any ticker's page, or one of the latest scan's top
// names (GET /v1/scans/latest, the same rows as the Scanner).
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import { TICKER_RE } from "@/components/AppShell";
import { Card, Empty, ErrorState, Freshness, Pill, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { pct, setupLabel } from "@/lib/format";

const TOP = 12;

export function StockSearchView() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [bad, setBad] = useState(false);
  const scan = useApi("stock-search-top", (signal) => unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: TOP } }, signal })));
  const open = (e: FormEvent) => {
    e.preventDefault();
    const t = q.trim().toUpperCase();
    if (!TICKER_RE.test(t)) {
      setBad(true);
      return;
    }
    router.push(`/stocks/${encodeURIComponent(t)}`);
  };
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Stock Intelligence</h1>
          <p className="cap">One ticker in depth: HSF Score and its parts, signals, why it ranks, risks, what to watch and its recent history.</p>
        </div>
      </section>
      <Card>
        <form className="row-actions" onSubmit={open} noValidate>
          <label className="field grow"><span>Ticker</span>
            <input value={q} maxLength={10} autoComplete="off" placeholder="e.g. NVDA" autoCapitalize="characters"
              aria-invalid={bad || undefined} aria-describedby={bad ? "stock-q-err" : undefined}
              onChange={(e) => { setQ(e.target.value); setBad(false); }} />
          </label>
          <button type="submit" className="btn btn-primary">Open</button>
        </form>
        {bad && <p id="stock-q-err" className="form-error" role="alert">Enter a ticker symbol like AAPL.</p>}
        <p className="cap">Any U.S. ticker opens. Scores and setups appear for names in the latest scan.</p>
      </Card>
      <Card title="Top of the latest scan" id="top" aside={scan.data?.scan_at ? <Freshness at={scan.data.scan_at} /> : undefined}>
        {scan.error && !scan.data ? <ErrorState error={scan.error} onRetry={scan.reload} what="the latest scan" />
          : !scan.data ? <Skeleton rows={6} label="Loading the latest scan" />
          : scan.data.setups.length === 0 ? <Empty title="No ranked setups in the latest scan." />
          : (
            <ul className="rows">
              {scan.data.setups.map((s) => (
                <li key={s.ticker} className="row">
                  <TickerLink ticker={s.ticker} />
                  <ScoreBadge score={s.score} />
                  {s.primary_setup && <Pill>{setupLabel(s.primary_setup)}</Pill>}
                  <span className="grow" />
                  <span className={`mono cap ${s.chg_pct == null ? "" : s.chg_pct >= 0 ? "up" : "down"}`}>{pct(s.chg_pct)}</span>
                </li>
              ))}
            </ul>
          )}
      </Card>
    </div>
  );
}
