"use client";

// The landing page's live preview: the latest market scan's top three ranked setups
// (GET /v1/public/top-setups), each linking to its public share page, and how many more a
// free account sees. Falls back to the sample card while loading or if the API is down.
import Link from "@/components/Link";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ScoreBadge } from "@/components/ui";
import { SampleSetupPreview } from "@/features/SampleSetupPreview";
import { useApi } from "@/hooks/useApi";
import { etTime, pct, setupLabel } from "@/lib/format";
import { track } from "@/lib/funnel";

type TopSetups = Schemas["PublicTopSetups"];

export function LiveSetupsPreview() {
  const res = useApi<TopSetups>("public-top-setups", (signal) => unwrap(publicApi.GET("/v1/public/top-setups", { signal })));
  const data = res.data;
  if (!data || !data.setups.length) return <SampleSetupPreview />;
  const more = Math.max(0, data.free_rows - data.setups.length);
  return (
    <aside className="card sample-preview live-preview" aria-labelledby="live-preview-title">
      <div className="row">
        <span className="eyebrow" id="live-preview-title"><span className="live-dot" aria-hidden="true" />Live · today&apos;s top setups</span>
        <span className="grow" />
        <span className="cap">Scan {etTime(data.scan_at)}</span>
      </div>
      <ol className="live-list">
        {data.setups.map((s, i) => (
          <li key={s.ticker} className="row">
            <span className="cap mono">{i + 1}</span>
            <Link href={`/share/${encodeURIComponent(s.ticker)}`} className="tk">{s.ticker}</Link>
            <span className="cap grow">{setupLabel(s.setup)}</span>
            <span className={`mono cap ${(s.change_pct ?? 0) >= 0 ? "up" : "down"}`}>{pct(s.change_pct, 1)}</span>
            <ScoreBadge score={s.hsf_score} />
          </li>
        ))}
      </ol>
      {more > 0 && <p className="body-sm">+{more} more ranked setups, with the evidence and risks behind each, free with an account.</p>}
      <Link href="/signup" className="btn btn-primary" onClick={() => track("primary_cta_click", "live_preview")}>
        See all of today&apos;s setups free
      </Link>
      <p className="cap">HSF Score ranks technical evidence. It is not a prediction or a recommendation.</p>
    </aside>
  );
}
