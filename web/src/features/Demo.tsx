"use client";

import Link from "next/link";
import { ResearchGlossary } from "@/components/ResearchGlossary";
import { useState } from "react";

import { Card, Pill, ResearchNotice, ScoreBar } from "@/components/ui";
import { SampleSetupPreview } from "@/features/SampleSetupPreview";
import { SAMPLE_SETUPS as SETUPS } from "@/lib/sampleResearch";
import type { SampleSetup } from "@/lib/sampleResearch";
import { PublicFooter, PublicHeader } from "@/features/Landing";

const WATCHLIST = [
  ["Needs attention", "NVDA", "HSF 88 · New strongest setup"],
  ["Strengthening", "AMD", "HSF 76 · Volume improving"],
  ["Fading", "SHOP", "HSF 38 · Left ranked list"],
];

function DemoScanner({ selected, onSelect }: { selected: string; onSelect: (stock: SampleSetup) => void }) {
  return (
    <div id="demo-scanner" tabIndex={-1}>
    <Card title="Demo Scanner" id="demo-scanner" aside="Sample ranked scan">
      <p className="cap">Choose a ticker to explore its score, supporting signals, and risks.</p>
      <div className="table-wrap">
        <table className="table demo-results" aria-label="Demo ranked setups">
          <thead><tr><th scope="col">Ticker</th><th scope="col">Setup</th><th scope="col">HSF Score</th><th scope="col" className="hide-narrow">Move</th><th scope="col" className="hide-narrow">RVOL</th></tr></thead>
          <tbody>{SETUPS.map((s) => (
            <tr key={s.ticker}>
              <td><a className="tk" href="#demo-stock" aria-label={`Review ${s.ticker} sample`} aria-current={selected === s.ticker ? "true" : undefined} onClick={() => onSelect(s)}>{s.ticker}</a></td>
              <td><Pill>{s.setup}</Pill></td>
              <td><ScoreBar score={s.score} /></td>
              <td className={`mono hide-narrow ${s.chg.startsWith("+") ? "up" : "down"}`}>{s.chg}</td>
              <td className="mono hide-narrow">{s.rvol}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <ResearchGlossary />
      <p className="cap">Next: in your account, save a stock with the watchlist button, then create an alert from its research page.</p>
      <p className="cap">Fixed sample data, not live market data. Explore freely; this tour does not place trades or create alerts.</p>
    </Card>
    </div>
  );
}

function DemoStock({ leader }: { leader: SampleSetup }) {
  return (
    <div id="demo-stock" tabIndex={-1}><Card title="Stock detail" id="demo-stock" aside={`${leader.ticker} sample`}>
      <div className="demo-stock">
        <div className="score-ring" style={{ "--p": `${leader.score}%` } as React.CSSProperties}><span>{leader.score}</span></div>
        <div className="stack-sm">
          <p className="strong">Why {leader.ticker} matters in this sample</p>
          <ul className="bullets">
            <li>HSF Score {leader.score}: {leader.note}</li>
            <li>Relative volume {leader.rvol}: participation is above its recent baseline.</li>
            <li>Gap {leader.gap}: {leader.risk}</li>
          </ul>
          <ResearchNotice compact />
        </div>
      </div>
    </Card></div>
  );
}

function DemoWatchlist() {
  return (
    <div id="demo-watchlist" tabIndex={-1}><Card title="Watchlist intelligence" id="demo-watchlist" aside="Sample activity">
      <ul className="rows">
        {WATCHLIST.map(([state, ticker, note]) => (
          <li key={ticker} className="row">
            <Pill tone={state === "Fading" ? "warn" : state === "Strengthening" ? "up" : "gold"}>{state}</Pill>
            <span className="tk">{ticker}</span>
            <span className="cap grow">{note}</span>
          </li>
        ))}
      </ul>
    </Card></div>
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
  const [selected, setSelected] = useState(SETUPS[0]!);
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack demo-page">
        <section className="hero hero-with-preview demo-intro">
          <div className="stack-sm">
            <p className="eyebrow">Explore HSF</p>
            <h1 className="hero-title">Explore a sample scan.</h1>
            <p className="hero-text">See why a stock ranks, review the evidence, and follow changes on a watchlist. No account needed.</p>
            <p className="cap" role="status">Sample data — not live.</p>
            <div className="row-actions">
              <a href="#demo-scanner" className="btn btn-primary">Explore the scanner</a>
              <Link href="/signup">Create a free account</Link>
            </div>
          </div>
          <SampleSetupPreview href="#demo-stock" onReview={() => setSelected(SETUPS[0]!)} />
        </section>

        <div className="demo-dashboard">
          <div className="demo-primary"><DemoScanner selected={selected.ticker} onSelect={setSelected} /><DemoStock leader={selected} /></div>
          <div className="demo-secondary"><DemoWatchlist /><DemoAlerts /></div>
        </div>

        <div className="tiles demo-kpis" aria-label="Demo market summary">
          <div className="card tile"><span className="cap">Ranked setups</span><p className="tile-value">42</p><span className="tile-delta up">Fresh sample</span></div>
          <div className="card tile"><span className="cap">Top HSF Score</span><p className="tile-value">88</p><span className="tile-delta up">NVDA</span></div>
          <div className="card tile"><span className="cap">Watchlist changes</span><p className="tile-value">3</p><span className="tile-delta">Attention list</span></div>
          <div className="card tile"><span className="cap">Alerts fired</span><p className="tile-value">2</p><span className="tile-delta">Sample events</span></div>
        </div>

      </main>
      <PublicFooter />
    </div>
  );
}
