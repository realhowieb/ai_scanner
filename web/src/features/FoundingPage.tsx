"use client";

// The public Founding 100 page (/founding-100): seats left, what each tier includes, the
// five-year math, the guarantee and the FAQ, readable before anyone signs in. The button
// goes to sign-up and then straight to checkout (/account/founder). Seats come from the
// server render and refresh from GET /v1/public/founding.
import Link from "@/components/Link";

import type { Schemas } from "@/api/client";
import { PublicFooter, PublicHeader, useLandingVisit } from "@/features/Landing";
import { track } from "@/lib/funnel";
import { useSignedIn } from "@/lib/signedIn";

import { FOUNDER_HREF, FiveYearMath, usePublicFounding } from "./FoundingBanner";

type PublicFounding = Schemas["PublicFounding"];

const dollars = (c: number) => `$${Math.round(c / 100).toLocaleString("en-US")}`;

/** Sign-up first for visitors; the founder page (checkout) right after. */
export const CLAIM_HREF = `/signup?next=${encodeURIComponent(FOUNDER_HREF)}`;

const INCLUDES: Record<string, { plan: string; lines: string[] }> = {
  founder_pro: {
    plan: "Pro",
    lines: ["Everything in Pro for five calendar years", "Your own Nasdaq and combined-market scans, email alerts and scan alerts",
      "The live Day Trader monitor, the earnings calendar and CSV export of Scanner results"],
  },
  founder_elite: {
    plan: "Premium",
    lines: ["Everything in Premium for five calendar years", "Ask HSF AI research, AI scan summaries and scheduled custom scans",
      "50 alerts (Premium has 25) and twice the daily AI questions"],
  },
};

export const FOUNDER_EXTRAS: string[] = [
  "A numbered founder seat (#001 to #100) and a founder badge",
  "After the five years: 50% off any yearly HSF plan, for as long as you keep it",
  "3 founder invites: friends you invite get 60 days of Pro",
  "Early access to new features, and a vote in the founder poll on what we build next",
  "Your seat on the Founders Wall, with a name only if you choose",
];

export const FAQ: [string, string][] = [
  ["Does it renew?", "No. It's one payment for five calendar years. Nothing renews and you are never charged again."],
  ["What if it isn't for me?", "You can ask for a full refund within 14 days of paying. If HSF ever shuts down during your five years, you get a prorated refund for the time left. The founding terms spell out both."],
  ["I already pay monthly or yearly.", "Cancel your plan in Account first. You can claim a seat once that plan has fully ended, so you never pay twice for the same months."],
  ["Can I move from Founder Pro to Founder Elite?", "Yes. The upgrade is $500, and you keep your founder number and your end date."],
  ["Is this financial advice?", "No. HSF is a research tool. It ranks and explains stock setups; it doesn't tell you what to buy or sell and makes no promise of returns."],
];

function Seats({ d }: { d: PublicFounding }) {
  return (
    <div className="founding-seats" aria-label="Seats left">
      {d.tiers.map((t) => (
        <div key={t.tier} className="founding-seat">
          <span className="strong">{t.label}</span>
          <span className="founding-seat-n mono">{t.remaining}</span>
          <span className="cap">of {t.capacity} left · {dollars(t.price_cents)} once</span>
        </div>
      ))}
    </div>
  );
}

export function FoundingPageView({ initial }: { initial: PublicFounding | null }) {
  useLandingVisit("founding_page");
  const live = usePublicFounding();
  const signedIn = useSignedIn();
  const d = live ?? initial;
  const claim = signedIn ? FOUNDER_HREF : CLAIM_HREF;
  const soldOut = !!d && d.remaining <= 0;
  const onSale = !!d && d.tiers.some((t) => t.on_sale);
  const cta = (surface: string, cls = "btn btn-primary") => (
    <Link href={claim} className={cls} onClick={() => track("primary_cta_click", `founding_page_${surface}`)}>
      {signedIn ? "Claim your seat" : "Create your account and claim a seat"}
    </Link>
  );
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        <section className="stack-sm" aria-labelledby="founding-title">
          <p className="eyebrow">Founding 100</p>
          <h1 id="founding-title" className="hero-title">100 founding memberships. One payment, five years.</h1>
          {!d ? (
            <p className="hero-text">The Founding 100 isn&apos;t open right now. <Link href="/pricing">See plans and pricing</Link></p>
          ) : soldOut ? (
            <p className="hero-text">All 100 seats are taken. Thank you, founders. <Link href="/founders">See the Founders Wall</Link> · <Link href="/pricing">See plans and pricing</Link></p>
          ) : (
            <>
              <p className="hero-text">{d.remaining >= 100 ? "All 100 seats are open." : `${d.remaining} of 100 seats left.`} Pay once, use HSF for five calendar years, and never see a renewal. Every seat is numbered, and there will never be more than 100.</p>
              <Seats d={d} />
              {onSale && <div className="row-actions">{cta("hero")}<Link href="/founding-100/terms" className="cap">Founding terms</Link></div>}
              <p className="cap">14-day full refund guarantee. Payments run through Stripe.</p>
            </>
          )}
        </section>

        {d && !soldOut && (
          <>
            <section aria-labelledby="math-h" className="card stack-sm">
              <h2 id="math-h" className="h2">Five years, compared</h2>
              <p className="body-sm">What the same five years cost on a monthly plan, against one founding payment.</p>
              <FiveYearMath tiers={d.tiers} />
            </section>

            <section aria-labelledby="tiers-h" className="stack-sm">
              <h2 id="tiers-h" className="h2">Pick your seat</h2>
              <div className="founder-grid">
                {d.tiers.map((t) => {
                  const inc = INCLUDES[t.tier];
                  return (
                    <article key={t.tier} className="card stack-sm" aria-labelledby={`tier-${t.tier}`}>
                      <div className="stack-xs">
                        <h3 id={`tier-${t.tier}`} className="h3">{t.label}</h3>
                        <p className="plan-price mono">{dollars(t.price_cents)} <span className="cap">once</span></p>
                        <p className="cap">{t.remaining} of {t.capacity} seats left</p>
                      </div>
                      {inc && <ul className="bullets">{inc.lines.map((l) => <li key={l}>{l}</li>)}</ul>}
                      {t.on_sale && t.remaining > 0 ? cta(t.tier, t.tier === "founder_pro" ? "btn btn-primary" : "btn") : <p className="cap">Not on sale right now.</p>}
                    </article>
                  );
                })}
              </div>
              <p className="cap">Founder Elite&apos;s planned extras, such as personal API access, are marked planned in the terms and aren&apos;t available yet. <Link href="/pricing">Compare every plan feature</Link></p>
            </section>

            <section aria-labelledby="extras-h" className="stack-sm">
              <h2 id="extras-h" className="h2">What every founder also gets</h2>
              <ul className="bullets">{FOUNDER_EXTRAS.map((l) => <li key={l}>{l}</li>)}</ul>
              <p className="cap"><Link href="/founders">See the Founders Wall</Link></p>
            </section>
          </>
        )}

        <section aria-labelledby="faq-h" className="stack-sm">
          <h2 id="faq-h" className="h2">Questions</h2>
          <div className="stack-xs">
            {FAQ.map(([q, a]) => (
              <details key={q} className="card founding-faq">
                <summary className="strong">{q}</summary>
                <p className="body-sm">{a}</p>
              </details>
            ))}
          </div>
          <p className="cap">Read the full <Link href="/founding-100/terms">founding terms</Link> and <Link href="/founding-100/refunds">refund policy</Link>.</p>
        </section>

        {d && onSale && !soldOut && <div className="row-actions">{cta("footer")}</div>}
        <p className="cap">HSF is a market-research tool, not financial advice. It makes no promise of returns.</p>
      </main>
      <PublicFooter />
    </div>
  );
}
