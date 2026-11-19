"use client";

// The public weekly recap (/weekly and /weekly/2026-W44): what HSF's scanner ranked most in a
// finished week, how the week before's top picks did over their 5 trading days (losers shown
// beside winners, linked to the full public track record), and the biggest earnings ahead.
import Link from "@/components/Link";

import type { Schemas } from "@/api/client";
import { TickerLogo } from "@/components/TickerLogo";
import { EmailSignup } from "@/features/EmailSignup";
import { FoundingStrip } from "@/features/FoundingBanner";
import { PublicFooter, PublicHeader, useLandingVisit } from "@/features/Landing";
import { setupLabel } from "@/lib/format";
import { track } from "@/lib/funnel";
import { weekRange } from "@/lib/weekRange";

type Recap = Schemas["PublicRecap"];
type Week = Schemas["RecapWeek"];

const DAY = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
const d = (iso: string) => new Date(`${iso}T12:00:00Z`);

const TIME: Record<string, string> = { bmo: "before open", amc: "after close" };

const pct = (v: number | null | undefined) => (v == null ? "—" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`);
const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

function LastWeek({ lw }: { lw: NonNullable<Recap["last_week"]> }) {
  const row = (p: (typeof lw.best)[number]) => (
    <tr key={`${p.day}-${p.ticker}`}>
      <th scope="row"><TickerLogo ticker={p.ticker} size={16} /> <span className="mono strong">{p.ticker}</span></th>
      <td className="cap">{DAY.format(d(p.day))}</td>
      <td className={`num mono ${tone(p.return_5d)}`}>{pct(p.return_5d)}</td>
      <td className={`num mono ${tone(p.excess_5d)}`}>{pct(p.excess_5d)}</td>
    </tr>
  );
  return (
    <section className="card stack-sm" aria-labelledby="lastweek-h">
      <h2 id="lastweek-h" className="h2">How last week&apos;s top picks did</h2>
      <p className="body-sm">
        The {lw.picks} daily top-10 picks from {weekRange(lw.start, lw.end)}, each over the 5 trading days after it was picked:
        average <span className={`mono ${tone(lw.avg_return)}`}>{pct(lw.avg_return)}</span> against <span className="mono">{pct(lw.avg_spy)}</span> for
        the S&amp;P 500 (<span className={`mono ${tone(lw.avg_excess)}`}>{pct(lw.avg_excess)}</span> difference). {lw.beat_rate != null && `${Math.round(lw.beat_rate * 100)}% beat the S&P 500.`}
      </p>
      <div className="grid2">
        {[["Best", lw.best], ["Worst", lw.worst]].map(([label, list]) => (
          <div key={label as string} className="table-wrap"><table className="table">
            <caption className="cap">{label as string} three</caption>
            <thead><tr><th scope="col">Stock</th><th scope="col">Picked</th><th scope="col" className="num">5 days</th><th scope="col" className="num">vs S&amp;P</th></tr></thead>
            <tbody>{(list as typeof lw.best).map(row)}</tbody>
          </table></div>
        ))}
      </div>
      <p className="cap">One week says very little about whether HSF has an edge. <Link href="/results">See every pick on the track record</Link>, losers included. No trading costs applied; not a recommendation.</p>
    </section>
  );
}

export function RecapBody({ recap }: { recap: Recap }) {
  return (
    <>
      <section className="card stack-sm" aria-labelledby="ranked-h">
        <h2 id="ranked-h" className="h2">Ranked most this week</h2>
        <p className="body-sm">
          Each trading day&apos;s first scan, its top 10 by HSF Score, grouped by stock. {recap.ranked_names} different
          stocks made a top 10 across {recap.scan_days} {recap.scan_days === 1 ? "day" : "days"}.
        </p>
        <div className="table-wrap"><table className="table">
          <thead><tr><th scope="col">Stock</th><th scope="col" className="num">Days in top 10</th>
            <th scope="col" className="num">Best HSF Score</th><th scope="col">Setup</th><th scope="col">First ranked</th></tr></thead>
          <tbody>{recap.ranked.map((r) => (
            <tr key={r.ticker}>
              <th scope="row">
                <Link href={`/share/${encodeURIComponent(r.ticker)}`} className="tk-link"><TickerLogo ticker={r.ticker} /> <span className="mono strong">{r.ticker}</span></Link>
                {r.name && <span className="cap"> {r.name}</span>}
              </th>
              <td className="num mono">{r.days}</td>
              <td className="num mono">{r.best_score}</td>
              <td>{r.setup ? setupLabel(r.setup) : "—"}</td>
              <td className="cap">{DAY.format(d(r.first_day))}</td>
            </tr>
          ))}</tbody>
        </table></div>
        <p className="cap">HSF Score (0–100) ranks how strongly the technical evidence lined up at scan time. It is not a probability of profit, and a ranking is not a recommendation to buy or sell.</p>
      </section>
      {recap.last_week && recap.last_week.picks > 0 && <LastWeek lw={recap.last_week} />}
      {recap.earnings && (
        <section className="card stack-sm" aria-labelledby="earn-h">
          <h2 id="earn-h" className="h2">Biggest earnings next week</h2>
          <p className="body-sm">The most-traded companies reporting each day, {weekRange(recap.earnings.start, recap.earnings.end)}.</p>
          <ul className="recap-days">
            {recap.earnings.days.map((day) => (
              <li key={day.date} className="tile">
                <p className="strong">{DAY.format(d(day.date))}</p>
                {day.names.length === 0 ? <p className="cap">No large reporters</p> : (
                  <ul className="stack-xs">{day.names.map((n) => (
                    <li key={n.ticker}><TickerLogo ticker={n.ticker} size={16} /> <span className="mono">{n.ticker}</span>
                      {n.time && TIME[n.time] && <span className="cap"> {TIME[n.time]}</span>}</li>
                  ))}</ul>
                )}
                {day.total > day.names.length && <p className="cap">+{day.total - day.names.length} more</p>}
              </li>
            ))}
          </ul>
          <p className="cap">The full calendar with HSF Scores is in the app. <Link href="/signup" onClick={() => track("primary_cta_click", "weekly_earnings")}>Create a free account</Link></p>
        </section>
      )}
    </>
  );
}

export function WeeklyRecapView({ recap, weeks }: { recap: Recap | null; weeks: Week[] }) {
  useLandingVisit("weekly");
  return (
    <div className="pub">
      <PublicHeader />
      <main id="main" className="pub-main stack">
        <header className="stack-xs">
          <p className="eyebrow">HSF weekly recap</p>
          <h1 className="h1">{recap ? `The week of ${weekRange(recap.start, recap.end)}` : "Weekly recap"}</h1>
          <p className="body-sm">What HSF&apos;s market scanner ranked, every week, recorded as it happened.</p>
        </header>
        {recap ? <RecapBody recap={recap} /> : (
          <section className="card stack-sm"><p className="body-sm">This recap isn&apos;t available. Recaps are published each weekend after the market closes on Friday.</p>
            {weeks.length > 0 && <p><Link href={`/weekly/${weeks[0]!.week}`}>Read the latest recap</Link></p>}</section>
        )}
        <FoundingStrip surface="weekly" />
        <EmailSignup surface="weekly" />
        <section className="card stack-sm">
          <h2 className="h3">See today&apos;s ranked setups</h2>
          <p className="body-sm">HSF scans thousands of U.S. stocks through the day. The Starter plan is free and shows the ranked Scanner, stock pages and watchlists.</p>
          <div className="row-actions">
            <Link href="/signup" className="btn btn-primary" onClick={() => track("primary_cta_click", "weekly")}>Create a free account</Link>
            <Link href="/how-hsf-works" className="btn">How HSF works</Link>
          </div>
        </section>
        {weeks.length > 1 && (
          <nav aria-labelledby="archive-h" className="stack-xs">
            <h2 id="archive-h" className="h3">Earlier weeks</h2>
            <ul className="bullets">{weeks.filter((w) => w.week !== recap?.week).map((w) => (
              <li key={w.week}><Link href={`/weekly/${w.week}`}>{weekRange(w.start, w.end)}</Link></li>
            ))}</ul>
          </nav>
        )}
      </main>
      <PublicFooter />
    </div>
  );
}
