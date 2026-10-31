"use client";

// The sign-in page's side panel on wide screens: what's waiting after sign-in, the latest
// scan's top three setups (GET /v1/public/top-setups, same as the landing page) and the
// Founding 100 seats while the offer is open. Hidden on phones by CSS.
import Link from "@/components/Link";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ScoreBadge } from "@/components/ui";
import { usePublicFounding } from "@/features/FoundingBanner";
import { useApi } from "@/hooks/useApi";
import { etTime, pct } from "@/lib/format";

type TopSetups = Schemas["PublicTopSetups"];

export function LoginAside() {
  const top = useApi<TopSetups>("public-top-setups", (signal) => unwrap(publicApi.GET("/v1/public/top-setups", { signal })));
  const founding = usePublicFounding();
  const setups = top.data?.setups ?? [];
  return (
    <aside className="auth-aside" aria-labelledby="auth-aside-h">
      <Link href="/" className="brand brand-lg" tabIndex={-1} aria-hidden="true">HSFinest<span>.AI</span></Link>
      <div className="stack-sm">
        <h2 id="auth-aside-h" className="auth-aside-h">Today&apos;s setups are ready.</h2>
        <p className="body-sm">Sign in for the full Scanner, your watchlists and your alerts.</p>
      </div>
      <ul className="auth-ticks">
        <li>Ranked setups with the evidence and risks behind each</li>
        <li>Morning Brief and evening wrap by email</li>
        <li>Price, score and earnings alerts</li>
      </ul>
      {setups.length > 0 && (
        <section className="card auth-aside-card" aria-labelledby="auth-top-h">
          <div className="row">
            <span className="eyebrow" id="auth-top-h"><span className="live-dot" aria-hidden="true" />Top setups right now</span>
            <span className="grow" />
            <span className="cap">Scan {etTime(top.data?.scan_at)}</span>
          </div>
          <ol className="live-list">
            {setups.slice(0, 3).map((s) => (
              <li key={s.ticker} className="row">
                <Link href={`/share/${encodeURIComponent(s.ticker)}`} className="tk">{s.ticker}</Link>
                <span className="grow" />
                <span className={`mono cap ${(s.change_pct ?? 0) >= 0 ? "up" : "down"}`}>{pct(s.change_pct, 1)}</span>
                <ScoreBadge score={s.hsf_score} />
              </li>
            ))}
          </ol>
        </section>
      )}
      {founding && founding.remaining > 0 && (
        <Link href="/pricing" className="auth-founding">★ {founding.remaining} of 100 Founding seats left</Link>
      )}
    </aside>
  );
}
