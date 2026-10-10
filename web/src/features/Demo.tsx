"use client";

import Link from "next/link";

import { Card, Pill, ResearchNotice, ScoreBar } from "@/components/ui";
import { PublicFooter, PublicHeader } from "@/features/Landing";

const SETUPS = [
  { ticker: "NVDA", setup: "Breakout", score: 88, last: "$188.40", chg: "+3.20%", gap: "+1.80%", rvol: "2.6x", note: "Strong HSF Score, rising trend, and heavy relative volume." },
  { ticker: "CRWD", setup: "Gapper", score: 81, last: "$412.10", chg: "+2.10%", gap: "+4.70%", rvol: "2.1x", note: "Gap strength with enough liquidity for follow-up research." },
  { ticker: "AMD", setup: "Golden cross", score: 76, last: "$241.32", chg: "+1.40%", gap: "+0.60%", rvol: "1.8x", note: "Improving trend and participation, but less explosive than the leaders." },
  { ticker: "APP", setup: "PreBreakout", score: 69, last: "$312.88", chg: "-0.30%", gap: "-0.10%", rvol: "1.3x", note: "Watchlist candidate: evidence is building, not confirmed." },
];

const WATCHLIST = [
  ["Needs attention", "NVDA", "HSF 88 · New strongest setup"],
  ["Strengthening", "AMD", "HSF 76 · Volume improving"],
  ["Fading", "SHOP", "HSF 38 · Left ranked list"],
];

function DemoScanner() {
  return (
    <Card title="Demo Scanner" id="demo-scanner" aside="Sample ranked scan">
      <div className="demo-table" role="table" aria-label="Demo ranked setups">
        <div className="demo-row demo-head" role="row">
          <span>Ticker</span><span>Setup</span><span>HSF Score</span><span className="hide-narrow">Move</span><span className="hide-narrow">RVOL</span>
        </div>
        {SETUPS.map((s) => (
          <div key={s.ticker} className="demo-row" role="row">
            <span className="tk">{s.ticker}</span>
            <span><Pill>{s.setup}</Pill></span>
            <ScoreBar score={s.score} />
            <span className={`mono hide-narrow ${s.chg.startsWith("+") ? "up" : "down"}`}>{s.chg}</span>
            <span className="mono hide-narrow">{s.rvol}</span>
          </div>
        ))}
      </div>
      <p className="cap">The real Scanner uses live scheduled scan data. This demo is fixed sample data for product review.</p>
    </Card>
  );
}

function DemoStock() {
  const leader = SETUPS[0]!;
  return (
    <Card title="Stock detail" id="demo-stock" aside="NVDA sample">
      <div className="demo-stock">
        <div className="score-ring" style={{ "--p": `${leader.score}%` } as React.CSSProperties}><span>{leader.score}</span></div>
        <div className="stack-sm">
          <p className="strong">Why NVDA matters in this sample</p>
          <ul className="bullets">
            <li>HSF Score {leader.score}: strongest ranked setup in the demo scan.</li>
            <li>Relative volume {leader.rvol}: participation is above its recent baseline.</li>
            <li>Gap {leader.gap}: momentum is visible, but extended moves can fade.</li>
          </ul>
          <ResearchNotice compact />
        </div>
      </div>
    </Card>
  );
}

function DemoWatchlist() {
  return (
    <Card title="Watchlist intelligence" id="demo-watchlist" aside="Main list">
      <ul className="rows">
        {WATCHLIST.map(([state, ticker, note]) => (
          <li key={ticker} className="row">
            <Pill tone={state === "Fading" ? "warn" : state === "Strengthening" ? "up" : "gold"}>{state}</Pill>
            <span className="tk">{ticker}</span>
            <span className="cap grow">{note}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function DemoAlerts() {
  return (
    <Card title="Alerts and AI research" id="demo-alerts" aside="Premium sample">
      <ul className="timeline">
        <li><span className="cap">9:45 AM ET</span> <span className="mono strong">NVDA</span> crossed HSF 85 with rising volume.</li>
        <li><span className="cap">10:05 AM ET</span> <span className="mono strong">SHOP</span> faded below the ranked-list floor.</li>
      </ul>
      <div className="notice" role="note">
        <p className="strong">Claude sample summary</p>
        <p className="body-sm">NVDA leads the sample scan on HSF Score and relative volume. CRWD and AMD are secondary research candidates. Commentary is based only on the demo scan and is not investment advice.</p>
      </div>
    </Card>
  );
}

export function DemoPage() {
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack demo-page">
        <section className="hero stack-sm">
          <p className="eyebrow">Frontend-only demo</p>
          <h1 className="hero-title">See HSF with polished sample data.</h1>
          <p className="hero-text">A guided product tour for screenshots, sales demos, and quick evaluation. No account, no writes, no live market dependency.</p>
          <div className="row-actions">
            <Link href="/signup" className="btn btn-primary">Create a free account</Link>
            <Link href="/pricing" className="btn">Compare plans</Link>
          </div>
          <p className="banner" role="status">Demo mode uses fixed sample data. It is not live market data and does not place trades or create alerts.</p>
        </section>

        <section className="priority-panel" aria-labelledby="demo-now">
          <div className="priority-copy">
            <p className="cap">What matters now</p>
            <h2 className="h2" id="demo-now">NVDA is the strongest setup in this sample scan.</h2>
            <p className="body-sm">Start with the ranked scanner, review the stock evidence, then watch changes on the sample watchlist.</p>
          </div>
          <div className="priority-steps">
            <a href="#demo-scanner" className="btn btn-primary">Review Scanner</a>
            <a href="#demo-stock" className="btn">Open stock detail</a>
            <a href="#demo-watchlist" className="btn">See watchlist</a>
          </div>
        </section>

        <div className="tiles demo-kpis" aria-label="Demo market summary">
          <div className="card tile"><span className="cap">Ranked setups</span><p className="tile-value">42</p><span className="tile-delta up">Fresh sample</span></div>
          <div className="card tile"><span className="cap">Top HSF Score</span><p className="tile-value">88</p><span className="tile-delta up">NVDA</span></div>
          <div className="card tile"><span className="cap">Watchlist changes</span><p className="tile-value">3</p><span className="tile-delta">Attention list</span></div>
          <div className="card tile"><span className="cap">Alerts fired</span><p className="tile-value">2</p><span className="tile-delta">Sample events</span></div>
        </div>

        <div className="brief-dashboard demo-dashboard">
          <div className="brief-center"><DemoScanner /><DemoStock /></div>
          <div className="brief-right"><DemoWatchlist /><DemoAlerts /></div>
        </div>
      </main>
      <PublicFooter />
    </div>
  );
}
