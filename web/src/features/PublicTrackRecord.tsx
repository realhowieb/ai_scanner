"use client";

// The public track record (/results): every day's top 10 HSF picks, frozen when the scan ran,
// with how each did over the next 5 trading days against the S&P 500. Losers are listed next
// to winners and the verdict is the API's own (a day-clustered bootstrap), worded plainly
// whatever it says. Picks appear only after their 5 days are over, so nothing current leaks.
import { useMemo, useState } from "react";

import type { Schemas } from "@/api/client";
import Link from "@/components/Link";
import { TickerLogo } from "@/components/TickerLogo";
import { EmailSignup } from "@/features/EmailSignup";
import { ConditionPill } from "@/features/MarketCondition";
import { PublicFooter, PublicHeader, useLandingVisit } from "@/features/Landing";
import { setupLabel } from "@/lib/format";
import { track } from "@/lib/funnel";

type Rec = Schemas["PublicRecord"];
type Pick = Schemas["RecordPick"];

const DAY = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
const day = (iso: string) => DAY.format(new Date(`${iso}T12:00:00Z`));

/** 0.012 -> "+1.20%" (the API sends fractions). */
export function pct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const p = v * 100;
  return `${p > 0 ? "+" : ""}${p.toFixed(digits)}%`;
}

const share = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");
const CONDITION_LABEL: Record<string, string> = { trending_up: "Trending up", choppy: "Choppy", risk_off: "Risk-off" };

const VERDICT_TITLE: Record<Rec["verdict"], string> = {
  no_edge_yet: "No proven edge yet",
  ahead: "Ahead of the S&P 500 so far",
  behind: "Behind the S&P 500 so far",
};

function Summary({ r }: { r: Rec }) {
  const s = r.summary;
  return (
    <section className="card stack-sm" aria-labelledby="sum-h">
      <p className="eyebrow">{s.first_day && s.last_day ? `${day(s.first_day)} to ${day(s.last_day)} · ${s.days} trading days` : "So far"}</p>
      <h2 id="sum-h" className="h2">{VERDICT_TITLE[r.verdict]}</h2>
      <p className="body-sm">{r.verdict_text}</p>
      <div className="record-stats">
        <div className="tile"><p className="cap">Picks scored</p><p className="mono strong">{s.picks.toLocaleString()}</p></div>
        <div className="tile"><p className="cap">Average 5-day return</p><p className={`mono strong ${tone(s.avg_return)}`}>{pct(s.avg_return)}</p></div>
        <div className="tile"><p className="cap">S&amp;P 500, same days</p><p className={`mono strong ${tone(s.avg_spy)}`}>{pct(s.avg_spy)}</p></div>
        <div className="tile"><p className="cap">Difference</p><p className={`mono strong ${tone(s.avg_excess)}`}>{pct(s.avg_excess)}</p>
          {s.excess_ci95 && <p className="cap">likely range {pct(s.excess_ci95.low)} to {pct(s.excess_ci95.high)}</p>}</div>
        <div className="tile"><p className="cap">Beat the S&amp;P 500</p><p className="mono strong">{share(s.beat_rate)}</p></div>
        <div className="tile"><p className="cap">Went up at all</p><p className="mono strong">{share(s.up_rate)}</p></div>
      </div>
      <p className="cap">
        &quot;Likely range&quot; is a 95% interval that resamples whole trading days, because picks on the same day tend to move
        together. We only call it ahead or behind when the whole range is on one side of zero and there are at least {r.min_days} trading days.
        {s.pending > 0 && ` ${s.pending} newer picks are still inside their 5 days and aren't scored yet.`}
        {s.unavailable > 0 && ` ${s.unavailable} had no price data afterwards and are left out.`}
      </p>
    </section>
  );
}

function ByCondition({ r }: { r: Rec }) {
  if (!r.by_condition.length) return null;
  return (
    <section className="card stack-sm" aria-labelledby="cond-h">
      <h2 id="cond-h" className="h2">By market condition</h2>
      <p className="body-sm">Setups behave differently when the market trends than when it chops around. Each pick is placed by the market condition at the close before it was picked.</p>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Results by market condition">
        <table className="table"><thead><tr><th scope="col">Market</th><th scope="col" className="num">Picks</th><th scope="col" className="num">Days</th>
          <th scope="col" className="num">Avg return</th><th scope="col" className="num">Avg vs S&amp;P 500</th><th scope="col" className="num">Beat S&amp;P 500</th></tr></thead>
          <tbody>{r.by_condition.map((g) => (
            <tr key={g.condition} className={g.days < 5 ? "row-muted" : undefined}>
              <th scope="row"><ConditionPill condition={g.condition} label={g.label} /></th>
              <td className="num mono">{g.picks}</td><td className="num mono">{g.days}</td>
              <td className={`num mono ${tone(g.avg_return)}`}>{pct(g.avg_return)}</td>
              <td className={`num mono ${tone(g.avg_excess)}`}>{pct(g.avg_excess)}</td>
              <td className="num mono">{share(g.beat_rate)}</td>
            </tr>
          ))}</tbody></table>
      </div>
      <p className="cap">Muted rows have fewer than 5 trading days: too few to read anything into. <Link href="/how-hsf-works#method-market">How market condition is decided</Link></p>
    </section>
  );
}

type Sort = "newest" | "best" | "worst";

function Picks({ picks, perDay }: { picks: Pick[]; perDay: number }) {
  const [sort, setSort] = useState<Sort>("newest");
  const rows = useMemo(() => {
    const out = [...picks];
    if (sort === "best") out.sort((a, b) => (b.return_5d ?? -Infinity) - (a.return_5d ?? -Infinity));
    if (sort === "worst") out.sort((a, b) => (a.return_5d ?? Infinity) - (b.return_5d ?? Infinity));
    return out;
  }, [picks, sort]);
  return (
    <section className="card stack-sm" aria-labelledby="picks-h">
      <h2 id="picks-h" className="h2">Every recent pick</h2>
      <p className="body-sm">The newest {picks.length} scored picks: each trading day&apos;s first scan, its top {perDay} by HSF Score (fewer on days when fewer score 40 or higher).</p>
      <div className="chips" role="group" aria-label="Sort picks">
        {(["newest", "best", "worst"] as const).map((k) => (
          <button key={k} type="button" className="chip" aria-pressed={sort === k} onClick={() => setSort(k)}>
            {k === "newest" ? "Newest" : k === "best" ? "Best first" : "Worst first"}
          </button>
        ))}
      </div>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Recent picks; scroll sideways for more columns">
        <table className="table"><thead><tr><th scope="col">Stock</th><th scope="col">Picked</th><th scope="col" className="num">HSF Score</th>
          <th scope="col" className="hide-narrow">Setup</th><th scope="col" className="num">5-day return</th><th scope="col" className="num">S&amp;P 500</th>
          <th scope="col" className="num">Difference</th><th scope="col" className="hide-narrow">Market</th></tr></thead>
          <tbody>{rows.map((p) => (
            <tr key={`${p.day}-${p.ticker}`}>
              <th scope="row"><Link href={`/share/${encodeURIComponent(p.ticker)}`} className="tk-link"><TickerLogo ticker={p.ticker} size={16} /> <span className="mono strong">{p.ticker}</span></Link></th>
              <td className="cap">{day(p.day)}</td>
              <td className="num mono">{p.score}</td>
              <td className="hide-narrow">{p.setup ? setupLabel(p.setup) : "—"}</td>
              <td className={`num mono ${tone(p.return_5d)}`}>{pct(p.return_5d, 1)}</td>
              <td className="num mono">{pct(p.spy_5d, 1)}</td>
              <td className={`num mono ${tone(p.excess_5d)}`}>{pct(p.excess_5d, 1)}</td>
              <td className="hide-narrow cap">{p.condition ? CONDITION_LABEL[p.condition] ?? p.condition : "—"}</td>
            </tr>
          ))}</tbody></table>
      </div>
    </section>
  );
}

export function PublicTrackRecordView({ record }: { record: Rec | null }) {
  useLandingVisit("results");
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        <header className="stack-xs">
          <p className="eyebrow">HSF track record</p>
          <h1 className="h1">Every top pick, and how it did</h1>
          <p className="body-sm">
            Each trading day, HSF saves up to {record?.top_per_day ?? 10} top setups (HSF Score {record?.min_score ?? 40} or higher) from its first scan the moment the scan runs. Five trading days later
            we record how each one did against the S&amp;P 500. Nothing is picked or removed afterwards, and losers stay on the list.
          </p>
        </header>
        {record ? (
          record.summary.picks === 0 ? <section className="card"><p className="body-sm">No picks have finished their 5 trading days yet. Check back soon.</p></section> : (
            <>
              <Summary r={record} />
              <ByCondition r={record} />
              <Picks picks={record.recent} perDay={record.top_per_day} />
            </>
          )
        ) : <section className="card"><p className="body-sm">The track record couldn&apos;t load just now. Please try again in a minute.</p></section>}
        <section className="card stack-sm" aria-labelledby="rules-h">
          <h2 id="rules-h" className="h3">How these numbers are made</h2>
          <ul className="bullets">
            <li>Picks: each trading day&apos;s first scheduled scan, its top {record?.top_per_day ?? 10} by HSF Score (score {record?.min_score ?? 40} or higher), one per stock. The same rule as the <Link href="/weekly">weekly recap</Link>.</li>
            <li>Returns: from the close on the day it was picked to the close 5 trading days later. The S&amp;P 500 (SPY) is measured over the same days.</li>
            <li>No trading costs, slippage, stops or targets are applied. Real trading results would differ, often for the worse.</li>
            <li>A pick is shown only after its 5 days are over, so this page never shows today&apos;s setups.</li>
            <li>Past results don&apos;t guarantee future ones. HSF is a research tool, not financial advice.</li>
          </ul>
        </section>
        <EmailSignup surface="results" />
        <section className="card stack-sm">
          <h2 className="h3">See today&apos;s ranked setups</h2>
          <p className="body-sm">The Starter plan is free and shows the ranked Scanner, stock pages and watchlists. Pro members get the full Track record with every score bucket, setup and holding period.</p>
          <div className="row-actions">
            <Link href="/signup" className="btn btn-primary" onClick={() => track("primary_cta_click", "results")}>Create a free account</Link>
            <Link href="/how-hsf-works" className="btn">How HSF works</Link>
          </div>
        </section>
      </main>
      <PublicFooter />
    </div>
  );
}
