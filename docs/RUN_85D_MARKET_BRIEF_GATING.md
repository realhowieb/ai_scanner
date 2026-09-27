# Run 85D — Market Brief Entitlement Gating

## Summary

- **Why:** Market Brief checked only two entitlements (admin diagnostics, and email
  delivery for Pro), so every Free account saw paid features the plans sell.
- **Now:** Market Brief follows the same entitlements as the Scanner.
- **Scoreboard:** the "Yesterday's brief — how it did" scoreboard is removed for everyone.
- **Unchanged:** HSF Scores are computed identically for every plan. Only what is
  displayed is gated.

## What Changed

| Section | Before | After | Entitlement (same as Scanner) |
|---|---|---|---|
| 🧠 PreBreakout picks (ranked model candidates) | Everyone | Premium; others see one line: "🧠 PreBreakout candidates · Premium unlocks this capability." | `can_early_breakout` (Scanner's "🔮 Early Breakout Candidates" tab) |
| Intelligent Alerts cards whose primary setup **is** PreBreakout (e.g. TWLO, A) | Everyone | Premium only | `can_early_breakout` |
| Standouts "prebreakout" tags (e.g. "TWLO — prebreakout + loser") | Everyone | Premium only (display copy without picks) | `can_early_breakout` |
| Claude-written market narrative | Everyone (a Claude call per snapshot per session) | Premium only | `can_ai_notes` |
| Per-opportunity "AI take" (a Claude call per opportunity) | Everyone | Premium only | `can_ai_notes` |
| 📊 Historical context per opportunity | Everyone | Pro and up | `can_track_record` |
| Historical research: flagged-signal outcomes (last 7 days) | Everyone | Pro and up | `can_track_record` |
| 📊 Yesterday's brief — how it did | Everyone (5 picks, one day, "avg +0.5%") | **Removed**: data function, renderer, section toggle and brief data key | — |
| "confirmation/context, not a predictive edge" | Repeated on every alert card | Once per section (Market Brief alerts, the opportunity feed, the watchlist intelligence feed) | — |
| "📧 Email me this brief" (Pro) | Included PreBreakout picks | Picks only for Premium | `can_early_breakout` |
| **Scheduled morning digest email** (Pro and up) | Included PreBreakout picks for every Pro+ recipient | Picks only for Premium and up | Premium tier check in `scheduler/morning_digest.py` |

**Still shown to Free** (the Discover tier):
- market header (regime, SPY/QQQ, breadth, leading sector);
- Top Opportunities with HSF Score and the rule-based "Why it ranked" and risk flags;
- what to watch next, sector leadership;
- Standouts from the non-PreBreakout lists;
- movers, gappers, setups (golden crosses, breakout scores);
- intelligent alerts from the other scanners;
- watchlist, open positions, fired alerts, catalysts.

Individual per-stock model probabilities remain under Model details everywhere (Run 79),
unchanged. What Premium sells is the ranked **candidate list**, and that is what is gated.

**Why scores aren't affected:** the brief's HSF Score uses PreBreakout picks as one of its
signals (`ui/opportunities.py`). Stripping picks from the data would have made the same
ticker score differently by plan. So scoring runs on the full data, and only the display
copy used by the picks section, Standouts and email drops the candidate list. A test
asserts identical HSF Scores for Free, Pro and Premium.

**Scheduled email change:** it's one guarded line in the per-recipient loop, deciding
which picks go into that recipient's email. No change to the schedule, the scan, scoring
or who receives the digest.

## Tests

**New: `tests/test_run85d_market_brief_gating.py` (9 tests).** They render the real Market
Brief page for Free, Pro and Premium from one realistic dataset, with Claude stubbed and
recorded:
- **Free:** no picks section or candidate list (one-line Premium note instead), no AI
  take, no historical context or outcomes, no prebreakout Standouts tags, and **zero
  Claude calls**.
- **Pro:** historical context and outcomes shown; no picks, no AI, zero Claude calls.
- **Premium:** picks, AI take (narrative and AI-take calls made) and historical context
  shown.
- **Everyone:** no scoreboard; HSF Scores identical across plans; the predictive-edge note
  appears at most once.
- **Email:** the emailed brief includes picks only for Premium; the scheduled digest gates
  picks to Premium; the scoreboard code is removed.
- **On the old code, 6 of the 9 fail.** The 3 that pass were already true: Premium sees
  everything, scores are identical, and this dataset has no scoreboard data (the
  source-level removal test covers that).

**Removed test:** `tests/test_market_brief.py::test_yesterday_performance_marks_picks_to_now`.
It tested `_yesterday_performance`, which was deleted along with the scoreboard feature.

**Results** (production-parity venv, outbound network blocked):

| Suite | Collected | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Full, `-X dev -W always` | 1971 | 1933 | 0 | 38 (FastAPI; covered by the billing job) |
| Lightweight CI env | 1971 | 1829 | 0 | 142 |
| `unittest discover` | 1928 run | OK | 0 | 38 |
| Billing contract | 111 | 110 | 0 | 1 |

**Named suites, all passing:**
- 85D: 9
- Market Brief + morning digest: 32
- Scheduler compatibility: 30
- Autonomy certification: 35
- 85C: 11
- Run 85: 22
- B2–B4: 24

**Other checks:** dependency audit clean; boot smoke rc 0; lint clean.

## Frozen Core

No change to scoring, ranking, the scan engine, models, research capture, maturation,
Gate U or the schedule. The only `scheduler/` edit is recipient-specific email content.
The certification test suite passes, and there's no lock or scanner change, so
re-certification isn't required.

## Follow-up (not in this run)

**Historical context also appears outside Market Brief without a Pro gate:**
- Scanner's results intelligence panel: `ui/results_intelligence.py:478`;
- Stock Intelligence: `ui/stock_intelligence.py:536`.

These are the same "70–79 range: X% positive outcome · n" lines. To be fully consistent
with this run they should also require `can_track_record`. Tracked as P1-32.
