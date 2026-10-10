"use client";

// The signed-out front door: what HSF does (the classic landing's copy, ui/product_copy.py),
// real product screenshots, and plans and prices from GET /v1/plans, the same source as
// the Billing page. No performance claims. Records the anonymous funnel events the
// classic app records, with the visitor's utm tags.
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { captureAttribution, track } from "@/lib/funnel";

type Plans = Schemas["Plans"];

const TAGLINE = "Turn the whole market into a short list.";
const POSITIONING = "Signal intelligence, not a prediction machine. HSF continuously scans thousands of tradable U.S. stocks and organizes noteworthy setups so you can focus your research.";
const SCORE_LINE = "HSF Score (0–100) ranks setups by how strongly the current technical evidence lines up. It is not a probability of profit or a prediction.";
const DISCLAIMER = "HSF is a market-research and decision-support tool, not financial advice. It does not recommend buying or selling any security.";
const PILLARS: [string, string][] = [
  ["Scan", "Continuously monitors thousands of tradable U.S. stocks through the trading day."],
  ["Rank", "Surfaces the setups that stand out right now."],
  ["Explain", "Shows the technical evidence behind each setup."],
  ["Track", "Shows how setups change through the session."],
];
const TRUST: [string, string][] = [
  ["Fresh market data", "Every result shows when it was scanned."],
  ["Transparent explanations", "Each setup lists the evidence that put it there."],
  ["Point-in-time scanning", "Results are recorded as the scanner saw them, never revised with hindsight."],
  ["Research-first methodology", "HSF measures its own methods forward before making claims about them."],
];
const VIEWS = [
  { id: "scanner", tab: "Scanner", title: "Ranked opportunities", text: "Thousands of symbols. One focused shortlist.",
    src: "/landing/scanner-showcase.webp", w: 1578, h: 997, alt: "HSF Scanner showing ranked opportunities, HSF Scores, statuses, and why the leading setup ranked" },
  { id: "day", tab: "Day Trader", title: "Stair-steppers", text: "Analyze smooth intraday momentum across multiple confirmation windows.",
    src: "/landing/day-trader-stair-stepper.webp", w: 1556, h: 1011, alt: "HSF Day Trader stair-stepper controls and ranked one-minute trend results" },
  { id: "brief", tab: "Market Brief", title: "Market Brief", text: "Know the market environment before evaluating the setup.",
    src: "/landing/market-brief-showcase.webp", w: 1536, h: 1024, alt: "HSF Market Brief showing market regime, index performance, breadth and alerts" },
] as const;

const VISIT_KEY = "hsf-landing-visit";

function SignupLink({ surface, children, className = "btn btn-primary" }: { surface: string; children: ReactNode; className?: string }) {
  return <Link href="/signup" className={className} onClick={() => track("primary_cta_click", surface)}>{children}</Link>;
}

export function PublicHeader() {
  return (
    <header className="pub-head">
      <Link href="/" className="brand">HSFinest<span>.AI</span></Link>
      <nav aria-label="Site" className="pub-nav">
        <Link href="/how-hsf-works">How it works</Link>
        <Link href="/demo">Demo</Link>
        <Link href="/pricing">Pricing</Link>
        <Link href="/login">Sign in</Link>
        <SignupLink surface="header" className="btn btn-primary btn-sm">Create free account</SignupLink>
      </nav>
    </header>
  );
}

export function PublicFooter() {
  return (
    <footer className="pub-foot">
      <p className="cap">{DISCLAIMER}</p>
      <p className="cap"><Link href="/how-hsf-works">How HSF works</Link> · <Link href="/demo">Demo</Link> · <Link href="/pricing">Pricing</Link> · <Link href="/login">Sign in</Link></p>
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
        <p className="eyebrow">Actual product views</p>
        <h2 id="showcase-title" className="h2">What HSF surfaces, and why</h2>
      </div>
      <div className="showcase-tabs" role="tablist" aria-label="Product views">
        {VIEWS.map((x) => (
          <button key={x.id} type="button" role="tab" id={`tab-${x.id}`} aria-selected={view === x.id} aria-controls="showcase-panel"
            className="showcase-tab" onClick={() => setView(x.id)}>{x.tab}</button>
        ))}
      </div>
      <figure id="showcase-panel" role="tabpanel" aria-labelledby={`tab-${v.id}`} className="showcase-panel">
        <figcaption className="stack-xs">
          <p className="strong">{v.title}</p>
          <p className="body-sm">{v.text}</p>
        </figcaption>
        <Image src={v.src} alt={v.alt} width={v.w} height={v.h} className="showcase-img" priority={v.id === "scanner"} unoptimized />
      </figure>
    </section>
  );
}

function cell(v: unknown) {
  if (typeof v === "number") return <span className="mono">{v}</span>;
  return v ? <span className="up" aria-label="Included">✓</span> : <span className="muted" aria-label="Not included">—</span>;
}

/** Plan cards, and with `table` the full comparison, from GET /v1/plans. */
export function PlansSection({ table = false }: { table?: boolean }) {
  const plans = useApi<Plans>("plans", (signal) => unwrap(publicApi.GET("/v1/plans", { signal })));
  if (plans.error && !plans.data) return <ErrorState error={plans.error} onRetry={plans.reload} what="plans and prices" />;
  if (!plans.data) return <Skeleton rows={6} label="Loading plans" />;
  const { tiers, rows } = plans.data;
  return (
    <div className="stack">
      <div className="plans">
        {tiers.map((t) => (
          <article key={t.id} className={`card plan${t.id === "pro" ? " plan-pick" : ""}`} aria-labelledby={`plan-${t.id}`}>
            <div className="stack-xs">
              <h3 id={`plan-${t.id}`} className="h2">{t.name}</h3>
              <p className="plan-price mono">{t.price}</p>
              {t.yearly_price && <p className="cap">or {t.yearly_price} (two months free)</p>}
              <p className="body-sm">{t.tagline}</p>
            </div>
            <ul className="bullets">
              {t.id !== "basic" && <li className="cap">Everything in {t.id === "pro" ? tiers[0]!.name : tiers[1]!.name}, plus:</li>}
              {t.highlights.map((h) => <li key={h}>{h}</li>)}
            </ul>
            <SignupLink surface={`plan_${t.id}`} className={t.id === "pro" ? "btn btn-primary" : "btn"}>
              {t.id === "basic" ? "Start free" : `Start free, upgrade to ${t.name}`}
            </SignupLink>
          </article>
        ))}
      </div>
      <p className="cap">Every account starts on {tiers[0]!.name}. Upgrade from Account after you verify your email; payments run through Stripe and you can cancel any time.</p>
      {table && (
        <div className="table-wrap">
          <table className="table compare">
            <caption className="sr-only">What each plan includes</caption>
            <thead>
              <tr><th scope="col">Feature</th>{tiers.map((t) => <th key={t.id} scope="col" className="num">{t.name}</th>)}</tr>
            </thead>
            <tbody>
              <tr><th scope="row" className="row-h">Price</th>{tiers.map((t) => <td key={t.id} className="num mono strong">{t.price}</td>)}</tr>
              {rows.map((r) => (
                <tr key={r.label}>
                  <th scope="row" className="row-h">{r.label}</th>
                  <td className="num">{cell(r.basic)}</td><td className="num">{cell(r.pro)}</td><td className="num">{cell(r.premium)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
        <section className="hero stack-sm">
          <p className="eyebrow">HSF AI Stock Scanner</p>
          <h1 className="hero-title">{TAGLINE}</h1>
          <p className="hero-text">{POSITIONING}</p>
          <div className="row-actions">
            <SignupLink surface="hero">Create a free account</SignupLink>
            <Link href="/demo" className="btn">View demo</Link>
            <Link href="/login" className="btn">Sign in</Link>
          </div>
          <p className="cap">Free plan, no card needed.</p>
        </section>

        <Showcase />

        <section aria-labelledby="what" className="stack-sm">
          <h2 id="what" className="h2">What HSF does</h2>
          <div className="tiles">
            {PILLARS.map(([t, d]) => <div key={t} className="card tile"><p className="strong">{t}</p><p className="body-sm">{d}</p></div>)}
          </div>
          <p className="cap">{SCORE_LINE}</p>
        </section>

        <section aria-labelledby="trust" className="stack-sm">
          <h2 id="trust" className="h2">Why you can trust what you see</h2>
          <div className="tiles">
            {TRUST.map(([t, d]) => <div key={t} className="card tile"><p className="strong">{t}</p><p className="body-sm">{d}</p></div>)}
          </div>
          <p className="cap"><Link href="/how-hsf-works">How HSF works, in full</Link></p>
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
          <p className="body-sm">Start free. Upgrade when you want your own scans, alerts by email, Day Trader, AI research or paper trading.</p>
        </section>
        <PlansSection table />
      </main>
      <PublicFooter />
    </div>
  );
}
