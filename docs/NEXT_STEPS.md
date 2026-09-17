# HSF — Working Notes & Next Steps

Rolling notes so context survives between sessions. Update as things change.

## Shipped recently (scanner universe + snapshot hardening)
- **PreBreakout 13.1% collapse** (`76f0b5a`) — legitimate isotonic calibration
  plateau; fixed by ranking plateau ties on the raw model score + a clarifying
  tooltip. Values unchanged.
- **Automation snapshot enrichment** (`ff241d6`) — export raw PreBreakout
  probability, apply canonical ranking to the snapshot, add model
  `feature_schema_version` fingerprints.
- **Live tradability filter** (`19cc391`, extended in `4e45ffc`) — drops
  delisted / non-tradable symbols (e.g. EA) via Alpaca's active+tradable assets
  at the scan chokepoint AND the regular engine path. Fails open. Surfaces
  `dropped_untradable` in the snapshot summary (~554/run observed).
- **Full-universe scanning** — `CRON_NASDAQ_LIMIT=10000` in the workflow; pre/
  post sessions now use the same SP500 + full-Nasdaq COMBO as the regular run
  (`ba5840e`). All 3 daily sessions scan ~4,771 symbols. Confirmed live.
- **Universe refresh** — `nasdaq.txt` rebuilt from nasdaqtrader.com (3,230 clean
  common stocks, `ae15e83`); `sp500.txt` rebuilt from SPDR SPY holdings (503)
  via `scripts/refresh_sp500.py` (stdlib xlsx parse, no new deps, `9d64d5b`).
- **Import-safety fixes** — `requests` optional in `data.fetch` (`d7e2a57`);
  `data/symbols.py` variable-width lookbehind regex fixed (`193c381`).

## Validation done
- Universe expansion materially changed output: ~77/100 postmarket candidates
  were previously unreachable (SP500-only); newly-scanned small-caps rank on
  merit (median rank 38.5 vs SP500 58.5).
- Model score distribution on new small-caps is **healthy / discriminative**,
  not out-of-distribution (raw fully unique, calibration escapes floor more
  often). NOTE: confirms distribution health, not accuracy — accuracy needs
  Run 24/25 outcome data to mature.

## Open threads (optional, not blocking)
1. **Automate the universe refresh** — `scripts/refresh_sp500.py` and the Nasdaq
   rebuild are manual. Could be a weekly workflow step. Low urgency (tradability
   filter handles delistings live; only new *listings* need a re-run).
2. **Model accuracy on small-caps** — distribution is healthy; accuracy is
   unproven until Run 24/25 outcomes accrue across the wider universe. Revisit
   once matured samples exist.
3. **PreBreakout recalibration** (post-V1 model maintenance) — the ~13% calibrated
   floor is well-evidenced; a leakage-safe low-range recalibration would narrow
   it. Deliberate DS task, not ad-hoc.
4. **LLM digest / snapshot consumer** (#3 from earlier) — deterministic digest
   builder → optional Claude narrative → delivery (email / in-app / .md / Slack)
   → cron step. Reads the local `latest_scan.json` the cron already writes.

## The recurring bigger item
Most recent work is scanner-**input** plumbing (valuable, now solid). The highest-
leverage unexamined area is the in-app experience (Market Brief, Stock
Intelligence first-run) and getting real users on the V1 RC. Let usage drive the
next round.

## Repo conventions (reminders)
- Branch: work on `dev`, ff-merge to `main`, push both.
- Gate: `.venv/bin/python -m pytest -q` + `python3.9 -m ruff check . --select E9,F,I`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
