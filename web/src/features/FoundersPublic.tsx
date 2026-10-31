"use client";

// Public Founding 100 pages: a shared founder card (/founders/card/<invite code>) and the
// Founders Wall (/founders). Neither ever shows an email; the wall shows a name only when
// the founder chose one.
import Link from "@/components/Link";

import type { Schemas } from "@/api/client";
import { Pill } from "@/components/ui";
import { PublicFooter, PublicHeader, useLandingVisit } from "@/features/Landing";
import { track } from "@/lib/funnel";
import { monthYear } from "@/lib/founders";

import { FOUNDER_HREF } from "./FoundingBanner";

type FounderCard = Schemas["PublicFounderCard"];
type FoundersWall = Schemas["FoundersWall"];

export function FounderCardView({ card }: { card: FounderCard | null }) {
  useLandingVisit("founder_card");
  const signup = card ? `/signup?ref=${encodeURIComponent(card.code)}` : "/signup";
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        {!card ? (
          <section className="card stack">
            <h1 className="h1">Founder card not found</h1>
            <p className="body-sm">This founder card isn&apos;t available. The founder may have turned off their badge.</p>
            <div className="row-actions"><Link href="/founders" className="btn">See the Founders Wall</Link><Link href="/" className="btn">What is HSF?</Link></div>
          </section>
        ) : (
          <>
            <section className={`founder-card${card.tier === "founder_elite" ? " founder-card-elite" : ""}`} aria-labelledby="card-h">
              <p className="eyebrow">Founding 100</p>
              <h1 id="card-h" className="h2">{card.label}</h1>
              <p className="founder-card-number" aria-label={`Founder number ${card.number}`}>#{card.number}</p>
              <dl className="kv">
                {card.since && <div><dt>Member since</dt><dd>{monthYear(card.since)}</dd></div>}
                {card.until && <div><dt>Valid to</dt><dd>{monthYear(card.until)}</dd></div>}
              </dl>
              {card.badge && <p><Pill tone="gold">{card.badge}</Pill></p>}
            </section>
            <section className="card stack-sm">
              <h2 className="h3">One of 100 founding members of HSFinest.AI</h2>
              <p className="body-sm">HSF scans the market and ranks stock setups with an HSF Score. Start free; join with this link and the founder who shared it gets credit.</p>
              <div className="row-actions">
                <Link href={signup} className="btn btn-primary" onClick={() => track("primary_cta_click", "founder_card")}>Create a free account</Link>
                <Link href="/founders" className="btn">See the Founders Wall</Link>
              </div>
            </section>
          </>
        )}
        <p className="cap">Research only, not financial advice.</p>
      </main>
      <PublicFooter />
    </div>
  );
}

export function FoundersWallView({ wall }: { wall: FoundersWall | null }) {
  useLandingVisit("founders_wall");
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        <header className="stack-xs">
          <p className="eyebrow">Founding 100</p>
          <h1 className="h1">Founders Wall</h1>
          {wall ? (
            <p className="body-sm">{wall.taken} of 100 seats taken. Every seat is listed by number; a name shows only when that founder chose to share one.</p>
          ) : (
            <p className="body-sm">The Founders Wall isn&apos;t open right now.</p>
          )}
        </header>
        {wall && wall.taken < 100 && (
          <div className="row-actions">
            <Link href={FOUNDER_HREF} className="btn btn-primary" onClick={() => track("primary_cta_click", "founding_wall")}>Claim a seat</Link>
            <Link href="/founding-100/terms" className="cap">Founding terms</Link>
          </div>
        )}
        {wall && (
          <ul className="wall" aria-label="Founder seats">
            {wall.seats.map((s) => (
              <li key={s.number} className={s.taken ? `wall-taken${s.tier === "founder_elite" ? " wall-elite" : ""}` : undefined}>
                <span className="wall-num">#{s.number}{s.label ? ` · ${s.label.replace("Founder ", "")}` : ""}</span>
                <span className="wall-name">{s.taken ? (s.name ?? "Founder") : "Open"}</span>
                {s.since && <span className="cap">Since {monthYear(s.since)}</span>}
              </li>
            ))}
          </ul>
        )}
        <p className="cap">Research only, not financial advice.</p>
      </main>
      <PublicFooter />
    </div>
  );
}
