"use client";

// Stock Intelligence home: open any ticker's page, or one of the latest scan's top
// names (GET /v1/scans/latest, the same rows as the Scanner).
import Link from "@/components/Link";
import { StockSearch } from "@/components/StockSearch";

import { api, unwrap } from "@/api/client";
import { Card, Empty, ErrorState, Freshness, Pill, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { pct, setupLabel } from "@/lib/format";

const TOP = 12;

export function StockSearchView() {
  const scan = useApi("stock-search-top", (signal) => unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: TOP } }, signal })));
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Stock Intelligence</h1>
          <Link href="/stocks/compare" className="btn btn-sm">Compare stocks</Link>
          <p className="cap">Search a company or ticker, or review a name from the latest ranked scan.</p>
        </div>
      </section>
      <Card title="Open a ticker" id="ticker-search"><StockSearch /></Card>
      <Card title="Top setups to review" id="top" aside={scan.data?.scan_at ? <Freshness at={scan.data.scan_at} stale={scan.data.stale} /> : undefined}>
        {scan.error && !scan.data ? <ErrorState error={scan.error} onRetry={scan.reload} what="the latest scan" />
          : !scan.data ? <Skeleton rows={6} label="Loading the latest scan" />
          : scan.data.setups.length === 0 ? <Empty title="No ranked setups in the latest scan." />
          : (
            <div className="stack-sm">
              <p className="cap">Each review opens the score, reasons, risks, watch items, chart and your lists or alerts for that ticker.</p>
              <ul className="rows review-queue">
                {scan.data.setups.map((s) => (
                  <li key={s.ticker} className="row">
                    <div className="stack-xs grow">
                      <div className="rc-top">
                        <TickerLink ticker={s.ticker} />
                        <ScoreBadge score={s.score} />
                        {s.primary_setup && <Pill>{setupLabel(s.primary_setup)}</Pill>}
                        <span className="cap review-meta">{s.status || "Ranked setup"} · {s.n_signals} confirming signal{s.n_signals === 1 ? "" : "s"}</span>
                      </div>
                    </div>
                    <span className="grow" />
                    <span className={`mono cap ${s.chg_pct == null ? "" : s.chg_pct >= 0 ? "up" : "down"}`}>{pct(s.chg_pct)}</span>
                    <Link href={`/stocks/${encodeURIComponent(s.ticker)}`} className="btn btn-sm">Review</Link>
                  </li>
                ))}
              </ul>
            </div>
          )}
      </Card>
    </div>
  );
}
