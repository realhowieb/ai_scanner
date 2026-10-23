"use client";

import Link from "@/components/Link";

import { Pill, ResearchNotice, ScoreBadge } from "@/components/ui";
import { PublicFooter, PublicHeader, useLandingVisit } from "@/features/Landing";
import { track } from "@/lib/funnel";
import { etTime, pct, price, setupLabel } from "@/lib/format";
import type { Schemas } from "@/api/client";

type PublicStock = Schemas["PublicStock"];

/** The signed-out page behind a shared stock link: what a preview promised, then sign-up. */
export function SharedStock({ stock: s, asked }: { stock: PublicStock | null; asked: string }) {
  useLandingVisit("share");
  const signup = s ? `/signup?next=${encodeURIComponent(`/stocks/${s.ticker}`)}` : "/signup";
  const cta = (label: string) => (
    <Link href={signup} className="btn btn-primary" onClick={() => track("primary_cta_click", "share")}>{label}</Link>
  );
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main share-main">
        {!s ? (
          <section className="card stack">
            <h1 className="h1 mono">{asked || "Stock"}</h1>
            <p className="body-sm">We couldn&apos;t find that stock, or HSF is busy right now. Try again in a moment.</p>
            <div className="row-actions">{cta("Create a free account")}<Link href="/" className="btn">What is HSF?</Link></div>
          </section>
        ) : (
          <section className="card stack" aria-labelledby="share-h">
            <div className="share-head">
              <div>
                <h1 className="h1 mono" id="share-h">{s.ticker}</h1>
                {s.name && <p className="cap">{s.name}</p>}
              </div>
              {s.price != null && (
                <p className="share-price mono"><span>{price(s.price)}</span>{" "}
                  <span className={(s.change_pct ?? 0) >= 0 ? "up" : "down"}>{pct(s.change_pct)}</span></p>
              )}
            </div>
            {s.ranked && s.hsf_score != null ? (
              <>
                <div className="share-score">
                  <ScoreBadge score={s.hsf_score} />
                  <div>
                    <p className="strong">In HSF&apos;s latest ranked setups</p>
                    <p className="cap">{s.setup && <Pill>{setupLabel(s.setup)}</Pill>} {s.scan_at ? `Scan ${etTime(s.scan_at)}` : ""}</p>
                  </div>
                </div>
                {s.reason && <p className="body-sm"><b>Why it ranks:</b> {s.reason}</p>}
                <ul className="share-locked" aria-label="With a free account">
                  {s.hidden_insights > 0 && <li>{s.hidden_insights} more reasons, signals and risks</li>}
                  <li>HSF Score history and price chart</li>
                  <li>Alerts when the score or setup changes</li>
                </ul>
                <div className="row-actions">{cta(`See the full ${s.ticker} analysis, free`)}</div>
              </>
            ) : (
              <>
                <p className="body-sm">{s.ticker} isn&apos;t in HSF&apos;s ranked setups right now. HSF scans the whole U.S. market several times a day and ranks the strongest setups.</p>
                <ul className="share-locked" aria-label="With a free account">
                  <li>Get an alert the moment {s.ticker} becomes a ranked setup</li>
                  <li>Today&apos;s top setups, with the reasons behind each score</li>
                </ul>
                <div className="row-actions">{cta("Create a free account")}<Link href="/demo" className="btn">Try the demo</Link></div>
              </>
            )}
            <p className="cap">Already have an account? <Link href={`/login?next=${encodeURIComponent(`/stocks/${s.ticker}`)}`}>Sign in</Link></p>
            <ResearchNotice compact />
          </section>
        )}
      </main>
      <PublicFooter />
    </div>
  );
}
