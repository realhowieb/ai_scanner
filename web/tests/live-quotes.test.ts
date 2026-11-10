import { expect, it } from "vitest";

import { withLiveQuotes } from "@/hooks/useLiveQuotes";

const rows = [
  { ticker: "AAA", last: 10, chg_pct: 1, gap_pct: 0.5, score: 90 },
  { ticker: "BBB", last: 20, chg_pct: -1, gap_pct: null, score: 80 },
  { ticker: "CCC", last: 30, chg_pct: 2, gap_pct: 1, score: 70 },
];

it("takes Last, Chg % and Gap % from live trades and keeps everything else", () => {
  const out = withLiveQuotes(rows, {
    AAA: { last: 11, chg_pct: 3.2, gap_pct: 0.4, source: "trade" },
    BBB: { last: 19, chg_pct: null, gap_pct: null, source: "trade" },
    CCC: { last: 31, chg_pct: 5, gap_pct: 2, source: "daily_bar" },
  });
  expect(out[0]).toEqual({ ticker: "AAA", scan_last: 10, last: 11, chg_pct: 3.2, gap_pct: 0.4, score: 90 });
  // No change or gap in the quote keeps the scan's.
  expect(out[1]).toEqual({ ticker: "BBB", scan_last: 20, last: 19, chg_pct: -1, gap_pct: null, score: 80 });
  // A daily-bar fallback is not a live trade.
  expect(out[2]).toBe(rows[2]);
});

it("returns the same rows when there are no quotes", () => {
  expect(withLiveQuotes(rows, {})).toBe(rows);
});
