// Fixed examples shared by the public preview and guided demo. Never live quotes.
export const SAMPLE_SETUPS = [
  { ticker: "NVDA", setup: "Breakout", score: 88, last: "$188.40", chg: "+3.20%", gap: "+1.80%", rvol: "2.6x", note: "Strong HSF Score, rising trend, and heavy relative volume.", risk: "An extended move can fade even when participation is strong." },
  { ticker: "CRWD", setup: "Gapper", score: 81, last: "$412.10", chg: "+2.10%", gap: "+4.70%", rvol: "2.1x", note: "Gap strength with enough liquidity for follow-up research.", risk: "A large opening gap can reverse during the session." },
  { ticker: "AMD", setup: "Golden cross", score: 76, last: "$241.32", chg: "+1.40%", gap: "+0.60%", rvol: "1.8x", note: "Improving trend and participation, but less explosive than the leaders.", risk: "A moving-average cross can lag a change in price direction." },
  { ticker: "APP", setup: "PreBreakout", score: 69, last: "$312.88", chg: "-0.30%", gap: "-0.10%", rvol: "1.3x", note: "Watchlist candidate: evidence is building, not confirmed.", risk: "An early setup may never develop into a breakout." },
];
export type SampleSetup = (typeof SAMPLE_SETUPS)[number];
