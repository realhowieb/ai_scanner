"use client";

// Today: how many of the latest scan's setups carry each Scanner signal (GET /v1/today
// setup_mix). Each row opens the Scanner with that signal's filter, so the count matches.
import Link from "@/components/Link";
import type { Schemas } from "@/api/client";
import { Card, Freshness, ScoreBadge, TickerLink } from "@/components/ui";

type Mix = Schemas["SetupMix"];

export function SetupMixCard({ mix, stale }: { mix: Mix | null | undefined; stale?: boolean | null }) {
  if (!mix || mix.total === 0 || !mix.rows.some((r) => r.count > 0)) return null;
  const most = Math.max(...mix.rows.map((r) => r.count), 1);
  return (
    <Card title="Setup mix" id="mix" source="ranking" aside={mix.scan_at ? <Freshness at={mix.scan_at} label="Full-market scan" stale={stale} /> : undefined}>
      <ul className="rows mix-rows">
        {mix.rows.map((r) => (
          <li key={r.signal} className="row mix-row">
            <Link href={`/scanner?signal=${encodeURIComponent(r.signal)}`} className="mix-label">{r.label}</Link>
            <div className="track" aria-hidden="true"><div className="fill" style={{ width: `${Math.round((r.count / most) * 100)}%` }} /></div>
            <span className="mono mix-count" aria-label={`${r.count} setup${r.count === 1 ? "" : "s"}`}>{r.count}</span>
            <span className="mix-top">
              {r.top_ticker ? <><span className="cap">Top</span> <TickerLink ticker={r.top_ticker} /> <ScoreBadge score={r.top_score} /></> : <span className="cap">None this scan</span>}
            </span>
          </li>
        ))}
      </ul>
      <p className="cap">
        How many of the latest scan&apos;s {mix.total} setups carry each signal. A name can carry more than one.
        Each row opens the Scanner filtered to that signal. A busy setup type isn&apos;t a forecast.
      </p>
    </Card>
  );
}
