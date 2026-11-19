import type { Metadata } from "next";

import { JsonLd } from "@/components/JsonLd";
import { publicMetadata } from "@/lib/seo";
import { faqSchema } from "@/lib/schema";
import Link from "@/components/Link";
import { PublicHeader, PublicFooter } from "@/features/Landing";

export const metadata: Metadata = publicMetadata("/how-hsf-works", "How HSF AI works",
  "What the HSF Score means, how fresh the data is, how trade plans handle entries, stops and size, how market conditions change results, and HSF's public track record. Educational research, not financial advice.");

// Public methodology for the current web app, readable signed in or out. Each entry is
// [anchor, heading, text]; the anchors are linked from the app (e.g. #method-market).
const SECTIONS: [string, string, string][] = [
  ["coverage", "Market coverage",
    "HSF scans a broad universe of tradable U.S.-listed stocks and ETFs on the major U.S. exchanges. The list is rebuilt from the market-data provider's tradable-asset list, so it changes as listings change. Preferred shares, SPAC units, warrants, rights and malformed symbols are excluded."],
  ["score", "What HSF Score means",
    "HSF Score is an opportunity-ranking score from 0 to 100. It orders setups by how strongly the current technical evidence lines up. It is not a probability of profit, not an expected return and not a prediction or guarantee. A higher score means a stronger current setup, not a better outcome."],
  ["execution", "From signal to trade: entries, stops and size",
    "A score on its own doesn't tell you where to get in, where you're wrong or how much to buy. On Pro, every ranked stock has a trade plan built from your own account size and risk per trade: an entry from the live price, a breakout trigger (the 20-day high), a don't-chase line (the scan price plus half the planned risk, so you can see when the cleanest entry has passed), a stop at half the stock's 20-day volatility (2 to 8%), targets at 1.5 and 3 times the risk, and a share count. It warns you when price has already run, when a stock trades thinly, when the bid-ask spread is wide, and when your position would be large next to a normal day's volume. You can set a live alert at the trigger or the stop in one tap. The plan is a risk-planning tool built from rules anyone can check, not a recommendation, and it doesn't model slippage or your broker's fills."],
  ["freshness", "How fresh the data is",
    "Full scans run every hour from 9:35 AM to 3:35 PM Eastern on trading days, plus 8:35 AM before the open and 4:35 and 5:35 PM after the close. Between scans, Scanner prices update about every minute. Day Trader refreshes live quotes, VWAP and relative volume every 15 seconds. Price, % move, relative volume and breakout-trigger alerts are checked about every 60 seconds and fire within about a minute. Score and setup alerts fire when a scan finds the change. Prices come from a single-exchange real-time feed (IEX through Alpaca), so they can differ slightly from the consolidated tape, and HSF doesn't show order-book depth (level 2). That makes HSF a good fit for swing trading and for building an intraday watchlist, and a poor fit for scalping or trading on split-second speed. Every result shows when it was scanned."],
  ["market", "Market conditions",
    "Setups that work in a steady uptrend fail more often when the market is choppy or falling. HSF labels the market with one transparent rule from the S&P 500 (SPY): Trending up when SPY is above its 20- and 50-day averages, the 20-day average is rising and 20-day volatility is under 25% a year; Risk-off when SPY is below both averages or volatility is 25% or more; Choppy otherwise. The label shows on the Scanner, the Market Brief and every trade plan, with what it means for a setup, and the Track record splits results by it so you can see how picks did in each kind of market. Past behaviour in a condition doesn't guarantee future results."],
  ["models", "Model details",
    "Some views provide supporting model outputs beneath HSF Score. Breakout score measures technical setup strength. PreBreakout setup probability estimates whether a quality setup will form in the next few trading days and subsequently meet its defined outcome. 5D outcome probability is the calibrated model probability of reaching +4% before -2% within five trading days. These are research estimates for defined events, not probabilities of profit, and they do not replace the HSF Score ranking. Breakout score supports the core ranking; PreBreakout and AI-assisted research are Premium capabilities."],
  ["why", "Why a stock appears",
    "Each result lists the technical and contextual evidence that put it there, such as relative volume, a price gap, trend over recent days, position against its recent high, strength against the S&P 500, and upcoming earnings. HSF explains the evidence; it does not tell you what to do with it."],
  ["record", "Public track record",
    "Every trading day HSF saves the top 10 setups from its first scan the moment it runs, and five trading days later records how each did against the S&P 500. The full list, losers included, is public on the track record page and updated daily, together with a plain verdict computed from the data. HSF is a young product: so far the results don't show a proven edge over the market, and the page says so. Treat any ranking as a starting point for your own research and check it on your own charts."],
  ["evaluation", "How HSF evaluates itself",
    "Live ranking asks what looks interesting now. Historical Research separately asks how similar saved HSF observations behaved afterward. Historical rankings use saved scores and prediction-time evidence rather than rescoring past picks with today's model. Changes to evaluation methods are versioned so incompatible results are not mixed. HSF records scanner observations as they happen and measures its methods forward, on data collected after the method was fixed. It makes no performance claims beyond the public track record, which shows the raw results as they are. Pro and Premium include the full Track record with every score bucket, setup, holding period and market condition."],
  ["advice", "Not financial advice",
    "HSF is a market-research and decision-support tool, not financial advice. It does not recommend buying or selling any security. Do your own research and consider your own circumstances before making any investment decision."],
];

export default function HowHsfWorks() {
  return (
    <div className="pub"><PublicHeader /><main id="main" className="doc">
      <JsonLd data={faqSchema(SECTIONS.map(([, h, body]) => [h, body] as [string, string]))} />
      <article className="card stack-sm">
        <h1 className="h1">How HSF works</h1>
        <p className="strong">Turn the whole market into a short list.</p>
        <p className="body-sm">HSF runs scheduled scans of a broad universe of tradable U.S.-listed stocks during the trading day. It ranks the setups that stand out, shows the technical evidence behind each one, and tracks how they change through the session. HSF is a research tool: it helps you decide what deserves a closer look, and leaves the decision to you.</p>
        <nav aria-label="On this page" className="chips">{SECTIONS.map(([id, h]) => <a className="chip" key={id} href={`#method-${id}`}>{h}</a>)}</nav>
        {SECTIONS.map(([id, h, body]) => (
          <section key={id} id={`method-${id}`} className="stack-sm doc-section">
            <h2 className="h2">{h}</h2>
            <p className="body-sm">{body}</p>
            {id === "record" && <p className="body-sm"><Link href="/results">See every pick and how it did</Link></p>}
            {id === "execution" && <p className="body-sm"><Link href="/demo#demo-plan">See a sample trade plan</Link></p>}
          </section>
        ))}
        <p className="cap"><Link href="/">← Back to HSF</Link></p>
      </article>
    </main><PublicFooter /></div>
  );
}
