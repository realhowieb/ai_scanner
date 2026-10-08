import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "How HSF works" };

// The classic app's methodology page (ui/methodology.py, ui/product_copy.py), word for
// word. Public, like the original: readable signed in or out.
const SECTIONS: [string, string][] = [
  ["Market coverage",
    "HSF scans a broad universe of tradable U.S.-listed stocks and ETFs on the major U.S. exchanges. The list is rebuilt from the market-data provider's tradable-asset list, so it changes as listings change. Preferred shares, SPAC units, warrants, rights and malformed symbols are excluded. Scheduled scans run several times each trading day."],
  ["What HSF Score means",
    "HSF Score is an opportunity-ranking score from 0 to 100. It orders setups by how strongly the current technical evidence lines up. It is not a probability of profit, not an expected return and not a prediction or guarantee. A higher score means a stronger current setup, not a better outcome."],
  ["Model details",
    "Some views provide supporting model outputs beneath HSF Score. Breakout score measures technical setup strength. PreBreakout setup probability estimates whether a quality setup will form in the next few trading days and subsequently meet its defined outcome. 5D outcome probability is the calibrated model probability of reaching +4% before -2% within five trading days. These are research estimates for defined events, not probabilities of profit, and they do not replace the HSF Score ranking. Breakout score supports the core ranking; PreBreakout and AI-assisted research are Premium capabilities."],
  ["Why a stock appears",
    "Each result lists the technical and contextual evidence that put it there, such as relative volume, a price gap, trend over recent days, position against its recent high, strength against the S&P 500, and upcoming earnings. HSF explains the evidence; it does not tell you what to do with it."],
  ["Data freshness",
    "Results are point-in-time scanner observations: they show what the scanner saw at the time shown, and they are never revised with hindsight. Check the scan time before acting on any result, especially outside market hours."],
  ["How HSF evaluates itself",
    "Live ranking asks what looks interesting now. Historical Research separately asks how similar saved HSF observations behaved afterward. HSF records scanner observations as they happen and measures its methods forward, on data collected after the method was fixed. HSF will not publish performance claims until that forward research supports them. Any historical figures shown in the app are labelled as historical research: they are descriptive and do not represent validated forward performance. Historical Research is available on Pro and Premium."],
  ["Not financial advice",
    "HSF is a market-research and decision-support tool, not financial advice. It does not recommend buying or selling any security. Do your own research and consider your own circumstances before making any investment decision."],
];

export default function HowHsfWorks() {
  return (
    <main className="doc">
      <article className="card stack-sm">
        <p className="brand brand-lg">HSFinest<span>.AI</span></p>
        <h1 className="h1">How HSF works</h1>
        <p className="strong">Turn the whole market into a short list.</p>
        <p className="body-sm">HSF scans a broad universe of tradable U.S.-listed stocks throughout the trading day. It ranks the setups that stand out, shows the technical evidence behind each one, and tracks how they change through the session. HSF is a research tool: it helps you decide what deserves a closer look, and leaves the decision to you.</p>
        {SECTIONS.map(([h, body]) => (
          <section key={h} className="stack-sm doc-section">
            <h2 className="h2">{h}</h2>
            <p className="body-sm">{body}</p>
          </section>
        ))}
        <p className="cap"><Link href="/today">← Back to HSF</Link></p>
      </article>
    </main>
  );
}
