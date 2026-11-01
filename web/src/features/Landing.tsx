"use client";
import { tabKey } from "@/lib/tabs";

// The signed-out front door: what HSF does (the classic landing's copy, ui/product_copy.py),
// real product screenshots, and plans and prices from GET /v1/plans, the same source as
// the Billing page. No performance claims. Records the anonymous funnel events the
// classic app records, with the visitor's utm tags.
import Image from "next/image";
import Link from "@/components/Link";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui";
import { BillingIntervalToggle, useBillingInterval } from "@/components/BillingInterval";
import { FoundingBanner, usePublicFounding } from "@/features/FoundingBanner";
import { LiveSetupsPreview } from "@/features/LiveSetupsPreview";
import { useApi } from "@/hooks/useApi";
import { captureAttribution, track } from "@/lib/funnel";
import { useSignedIn } from "@/lib/signedIn";
import { LegalLinks } from "@/components/LegalLinks";

type Plans = Schemas["Plans"];

export const HEADLINE = "Find Market Opportunities with AI-Powered Intelligence";
export const SUBHEADLINE = "Discover momentum, analyze technical signals, and monitor market activity from one intelligent platform.";
const POSITIONING = "Signal intelligence, not a prediction machine. HSF continuously scans thousands of tradable U.S. stocks and organizes noteworthy setups so you can focus your research.";
const SCORE_LINE = "HSF Score (0–100) ranks setups by how strongly the current technical evidence lines up. It is not a probability of profit or a prediction.";
const DISCLAIMER = "HSF is a market-research and decision-support tool, not financial advice. It does not recommend buying or selling any security.";
// Each one is a working part of the app today (no roadmap items).
export const FEATURES: [string, string][] = [
  ["AI Stock Scanner", "Scheduled scans of thousands of tradable U.S. stocks rank the setups that stand out, each with an HSF Score from 0 to 100."],
  ["Market Intelligence", "The Market Brief shows indexes, breadth, sectors, pre-market movers and the earnings calendar before you look at a single setup."],
  ["Technical Analysis", "Interactive price charts with moving averages, RSI and volume, plus the evidence behind every score: relative volume, gaps, trend and strength against the S&P 500."],
  ["Watchlists", "Save the stocks you follow, see how they move against the S&P 500 and get alerts when a price, score or setup changes."],
  ["Momentum and Breakout Discovery", "Day Trader ranks intraday movers, and early breakout and steady stair-step setups flag stocks while they are still building."],
  ["Historical Signal Tracking", "Every result is recorded as the scanner saw it. Score history and the Track record show how past setups developed, never revised with hindsight."],
];
const HOW: [string, string][] = [
  ["Discover stocks", "Start from the ranked Scanner, the Market Brief or your own custom scan."],
  ["Evaluate market signals", "Open any stock to see its chart, HSF Score and the evidence and risks behind it."],
  ["Monitor opportunities", "Save it to a watchlist and set an alert, so HSF tells you when something changes."],
];
const TRUST: [string, string][] = [
  ["Fresh market data", "Every result shows when it was scanned."],
  ["Transparent explanations", "Each setup lists the evidence that put it there."],
  ["Point-in-time scanning", "Results are recorded as the scanner saw them, never revised with hindsight."],
  ["Research-first methodology", "HSF measures its own methods forward before making claims about them."],
  ["Estimates are not outcomes", "Model outputs such as breakout or outcome probabilities are research estimates for defined events. They are not profits, returns or trading results."],
];
const VIEWS = [
  { id: "scanner", tab: "Scanner", title: "Ranked opportunities", text: "Thousands of symbols. One focused shortlist.",
    src: "/landing/scanner-current.jpg", w: 1425, h: 1451, alt: "HSF Scanner with sample data showing ranked opportunities, HSF Scores, statuses, and why the leading setup ranked" },
  { id: "day", tab: "Day Trader", title: "Day Trader", text: "Review intraday candidates, scores and market context.",
    src: "/landing/day-current.jpg", w: 1425, h: 1384, alt: "Current HSF Day Trader interface with sample intraday candidates and scores" },
  { id: "brief", tab: "Market Brief", title: "Market Brief", text: "Know the market environment before evaluating the setup.",
    src: "/landing/brief-current.jpg", w: 1425, h: 1844, alt: "Current HSF Market Brief with sample indexes, breadth, sectors and earnings" },
] as const;

const VISIT_KEY = "hsf-landing-visit";

function SignupLink({ surface, children, className = "btn btn-primary" }: { surface: string; children: ReactNode; className?: string }) {
  return <Link href="/signup" className={className} onClick={() => track("primary_cta_click", surface)}>{children}</Link>;
}

export function PublicHeader() {
  const signedIn = useSignedIn();
  return (
    <header className="pub-head">
      <Link href="/" className="brand">HSFinest<span>.AI</span></Link>
      <nav aria-label="Site" className="pub-nav">
        <Link href="/how-hsf-works">How it works</Link>
        <Link href="/demo">Demo</Link>
        <Link href="/pricing">Pricing</Link>
        {signedIn ? <Link href="/today" className="btn btn-primary btn-sm">Open HSF</Link> : <>
          <Link href="/login">Sign in</Link>
          <SignupLink surface="header" className="btn btn-primary btn-sm">Create free account</SignupLink>
        </>}
      </nav>
    </header>
  );
}

export function PublicFooter() {
  const signedIn = useSignedIn();
  const founding = usePublicFounding();
  return (
    <footer className="pub-foot">
      <p className="cap">{DISCLAIMER}</p>
      <p className="cap"><Link href="/how-hsf-works">How HSF works</Link> · <Link href="/demo">Demo</Link> · <Link href="/pricing">Pricing</Link> · {signedIn ? <Link href="/today">Open HSF</Link> : <Link href="/login">Sign in</Link>}</p>
      <p className="cap"><LegalLinks refunds={!!founding} /></p>
    </footer>
  );
}

/** Record where this visitor came from and one landing visit per browser session. */
export function useLandingVisit(surface: string) {
  useEffect(() => {
    captureAttribution();
    try {
      if (sessionStorage.getItem(VISIT_KEY)) return;
      sessionStorage.setItem(VISIT_KEY, "1");
    } catch {
      /* count it anyway */
    }
    track("landing_visit", surface);
  }, [surface]);
}

function Showcase() {
  const [view, setView] = useState<(typeof VIEWS)[number]["id"]>("scanner");
  const v = VIEWS.find((x) => x.id === view)!;
  return (
    <section className="card flush showcase" aria-labelledby="showcase-title">
      <div className="showcase-head">
        <p className="eyebrow">Current product views</p>
        <h2 id="showcase-title" className="h2">What HSF surfaces, and why</h2>
      </div>
      <div className="showcase-tabs" role="tablist" onKeyDown={tabKey} aria-label="Product views">
        {VIEWS.map((x) => (
          <button key={x.id} type="button" role="tab" tabIndex={view === x.id ? 0 : -1} id={`tab-${x.id}`} aria-selected={view === x.id} aria-controls="showcase-panel"
            className="showcase-tab" onClick={() => setView(x.id)}>{x.tab}</button>
        ))}
      </div>
      <figure id="showcase-panel" role="tabpanel" aria-labelledby={`tab-${v.id}`} className="showcase-panel">
        <figcaption className="stack-xs">
          <p className="strong">{v.title}</p>
          <p className="body-sm">{v.text}</p>
        </figcaption>
        {/* Use a separately captured mobile layout on phones; retain the full desktop capture. */}
        <picture>
          <source media="(max-width: 760px)" srcSet={v.src.replace("-current.jpg", "-mobile-current.jpg")} />
          <Image src={v.src} alt={v.alt} width={v.w} height={v.h} className="showcase-img" priority={v.id === "scanner"} unoptimized />
        </picture>
        <p className="cap">Current interface with illustrative sample data, not live prices. <a href={v.src} target="_blank" rel="noopener">Open full screenshot</a> · <Link href="/demo">Try the interactive demo</Link></p>
      </figure>
    </section>
  );
}

function cell(v: unknown) {
  if (typeof v === "number") return <span className="mono">{v}</span>;
  return v ? <span className="up" aria-label="Included">✓</span> : <span className="muted" aria-label="Not included">—</span>;
}

export function annualValue(monthly: string, yearly: string): string | null {
  const amount = (s: string) => s.startsWith("$") ? Number(s.match(/^\$([\d,.]+)/)?.[1]?.replaceAll(",", "")) : NaN;
  const m = amount(monthly), y = amount(yearly);
  if (!(m > 0 && y > 0 && y < m * 12)) return null;
  return `$${(y / 12).toFixed(2)}/mo equivalent · save ${Math.round((1 - y / (m * 12)) * 100)}% vs monthly`;
}

function summaryHighlights(id: string, highlights: string[]): string[] {
  if (id === "basic") return highlights.slice(0, 5);
  const priority = id === "pro" ? /^(\d+ alerts?$)|email|nasdaq|day trader|historical/i : /^(\d+ alerts?$)|AI |early breakout|full-market|alpaca/i;
  return [...highlights.filter((h) => priority.test(h)), ...highlights.filter((h) => !priority.test(h))].slice(0, 5);
}

/** Plan cards, and with `table` the full comparison, from GET /v1/plans. */
export function PlansSection({ table = false }: { table?: boolean }) {
  const [interval] = useBillingInterval();
  const [mobilePlan, setMobilePlan] = useState("pro");
  const signedIn = useSignedIn();
  const plans = useApi<Plans>("plans", (signal) => unwrap(publicApi.GET("/v1/plans", { signal })));
  if (plans.error && !plans.data) return <ErrorState error={plans.error} onRetry={plans.reload} what="plans and prices" />;
  if (!plans.data) return <Skeleton rows={6} label="Loading plans" />;
  const { tiers, rows } = plans.data;
  const price = (tier: Plans["tiers"][number]) => interval === "year" && tier.yearly_price ? tier.yearly_price : tier.price;
  return (
    <div className="stack">
      <BillingIntervalToggle />
      <div className="plans">
        {tiers.map((t) => (
          <article key={t.id} className={`card plan${t.id === "pro" ? " plan-pick" : ""}`} aria-labelledby={`plan-${t.id}`}>
            <div className="stack-xs">
              <h3 id={`plan-${t.id}`} className="h2">{t.name}</h3>
              <p className="plan-price mono">{price(t)}</p>
              {t.yearly_price && <p className="cap">{interval === "month"
                ? `or ${t.yearly_price} with yearly billing` : `${t.price} with monthly billing`}</p>}
              {interval === "year" && t.id !== "basic" && !t.yearly_price && <p className="cap">Yearly pricing is not available for this plan.</p>}
              {interval === "year" && t.yearly_price && annualValue(t.price, t.yearly_price) && <p className="cap">{annualValue(t.price, t.yearly_price)}</p>}
              <p className="body-sm">{t.tagline}</p>
            </div>
            <ul className="bullets">
              {t.id !== "basic" && <li className="cap">Everything in {t.id === "pro" ? tiers[0]!.name : tiers[1]!.name}, plus:</li>}
              {summaryHighlights(t.id, t.highlights).map((h) => <li key={h}>{h}</li>)}
            </ul>
            {t.highlights.length > 5 && <details><summary>All plan features</summary><ul className="bullets">{t.highlights.filter((h) => !summaryHighlights(t.id, t.highlights).includes(h)).map((h) => <li key={h}>{h}</li>)}</ul></details>}
            {t.id === "pro" && <p className="cap">Highlighted for custom scans and intraday research.</p>}
            {signedIn
              ? <Link href={t.id === "basic" ? "/today" : "/account#plan"} className={t.id === "pro" ? "btn btn-primary" : "btn"}>
                {t.id === "basic" ? "Open HSF" : `Upgrade to ${t.name} in Account`}
              </Link>
              : <SignupLink surface={`plan_${t.id}`} className={t.id === "pro" ? "btn btn-primary" : "btn"}>
                {t.id === "basic" ? "Start free" : `Start free, upgrade to ${t.name}`}
              </SignupLink>}
          </article>
        ))}
      </div>
      <p className="cap">{signedIn ? "You're signed in, so change your plan from Account." : `Every account starts on ${tiers[0]!.name}. Upgrade from Account after you verify your email.`} Payments run through Stripe and you can cancel any time.</p>
      {table && (
        <><div className="only-narrow card stack-sm"><label className="field">Compare a plan<select value={mobilePlan} onChange={(e) => setMobilePlan(e.target.value)}>{tiers.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}</select></label>
          <dl className="kv">{rows.map((r) => <div key={r.label}><dt>{r.label}</dt><dd>{cell(r[mobilePlan as "basic" | "pro" | "premium"])}</dd></div>)}</dl>
        </div><div className="table-wrap only-wide">
          <table className="table compare">
            <caption className="sr-only">What each plan includes</caption>
            <thead>
              <tr><th scope="col">Feature</th>{tiers.map((t) => <th key={t.id} scope="col" className="num">{t.name}</th>)}</tr>
            </thead>
            <tbody>
              <tr><th scope="row" className="row-h">Price</th>{tiers.map((t) => <td key={t.id} className="num mono strong">{price(t)}</td>)}</tr>
              {rows.map((r) => (
                <tr key={r.label}>
                  <th scope="row" className="row-h">{r.label}</th>
                  <td className="num">{cell(r.basic)}</td><td className="num">{cell(r.pro)}</td><td className="num">{cell(r.premium)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div></>
      )}
    </div>
  );
}

export function Landing() {
  useLandingVisit("landing");
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        <section className="hero hero-with-preview">
          <div className="stack-sm">
          <p className="eyebrow">HSF AI · Discover opportunities. Understand the signals. Trade with intelligence.</p>
          <h1 className="hero-title">{HEADLINE}</h1>
          <p className="hero-text">{SUBHEADLINE}</p>
          <div className="row-actions">
            <SignupLink surface="hero">Try it free</SignupLink>
            <Link href="/how-hsf-works" className="btn">See How It Works</Link>
            <Link href="/login" className="hero-signin">Sign in</Link>
          </div>
          <p className="cap">Starter plan is free forever, no card needed. Takes under a minute.</p>
          </div>
          <LiveSetupsPreview />
        </section>

        <FoundingBanner surface="landing" />

        <section aria-labelledby="features" className="stack-sm">
          <h2 id="features" className="h2">What you can do with HSF</h2>
          <p className="body-sm">{POSITIONING}</p>
          <div className="feature-grid">
            {FEATURES.map(([t, d]) => <div key={t} className="card tile"><p className="strong">{t}</p><p className="body-sm">{d}</p></div>)}
          </div>
        </section>

        <Showcase />

        <section aria-labelledby="how" className="stack-sm">
          <h2 id="how" className="h2">How it works</h2>
          <ol className="steps">
            {HOW.map(([t, d]) => <li key={t}><b>{t}.</b> {d}</li>)}
          </ol>
        </section>

        <section aria-labelledby="trust" className="stack-sm">
          <h2 id="trust" className="h2">Why you can trust what you see</h2>
          <div className="feature-grid">
            {TRUST.map(([t, d]) => <div key={t} className="card tile"><p className="strong">{t}</p><p className="body-sm">{d}</p></div>)}
          </div>
          <p className="cap">{SCORE_LINE} <Link href="/how-hsf-works">How HSF works, in full</Link></p>
        </section>

        <section aria-labelledby="plans" className="stack-sm" id="pricing">
          <h2 id="plans" className="h2">Plans</h2>
          <PlansSection />
          <p className="cap"><Link href="/pricing">Compare every feature</Link></p>
        </section>
      </main>
      <PublicFooter />
    </div>
  );
}

export function PricingPage() {
  useLandingVisit("pricing");
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        <section className="stack-xs">
          <h1 className="h1">Plans and pricing</h1>
          <p className="body-sm">Start free with the ranked Scanner, stock pages, watchlists and your own S&amp;P 500 scans. Upgrade for Nasdaq and full-market scans, email alerts, Day Trader, the Track record, CSV export, AI research or paper trading.</p>
        </section>
        <FoundingBanner surface="pricing" />
        <PlansSection table />
      </main>
      <PublicFooter />
    </div>
  );
}
