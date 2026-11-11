# Cold paths and alert isolation — November 11, 2026

## Score history

The public history window is built with one slim batch query for missing scan
projections, rather than reading each full JSON payload and its version in
separate round trips. Model vectors, charts and account data are excluded at
the database boundary. The output still uses the Scanner's canonical scorer;
plan redaction is applied when forming each user's response.

`scan_score_cache` is disposable derived data, created lazily with the normal
schema-once helper. It retains at most 96 projections across all scorer methods.
Each projection carries the source run's timestamp/count stamp. Its method key
includes the canonical score version and a digest of scorer/consolidation code,
so unrelated deploys reuse it while scoring-code changes invalidate it. No
source runs or research observations are updated. Cache failure falls back to
rebuilding the slim projections. Unreadable or concurrently rewritten runs
remain gaps and are retried; successful points remain available.

Only the two most recent windows are held in API memory. Startup stock-page
warming and the existing five-minute ops pass prepare the window without
requiring a visitor to request it. The ops pass therefore catches new scans
within its normal polling cadence, not through a new scheduler callback.

## Real-time alerts and expected missing symbols

Malformed syntax is filtered, class-share spellings are mapped, and a symbol
explicitly rejected by the provider is removed before retrying the other
symbols. Unnamed HTTP 400/422 validation failures use bounded subdivision (at
most 24 calls per 100-symbol chunk). Auth/quota failures stop the pass; network
and server errors do not trigger subdivision. No alert settings or saved
thresholds are changed.

Acceptance limit tests use a real ticker, SPY, with distinct unreachable
thresholds; they no longer introduce ZQ00-style tickers into the live worker.
The unknown-ticker stock-page test remains. Yahoo history requests raise their
expected missing-price/timezone exceptions rather than logging ERROR messages;
those outcomes retain their normal skip/missing-cache behavior. Other failures
still warn. There is no blanket log-level suppression.

## Brief memory

Scheduled scans already save calibrated/raw PreBreakout values and provenance.
The API Brief now selects its picks from those values, preserving raw-score tie
breaks and saved model evidence. It does not import/load the model to rescore
on first open. If a scan has no saved probabilities, no picks are invented and
there is no API-side inference fallback. The standalone email digest's default
scoring behavior is unchanged. The existing per-scan small-pick memo remains.

`scripts/diag_brief_saved_picks.py` compares the saved and rescored paths in
separate fresh processes and reports only aggregate RSS, elapsed time, module
presence and output digests. It performs no fitting or model promotion. The
runner's RSS is not Render container working-set memory. Reconstructed fresh
scores can differ from saved prediction-time scores; the script reports this
rather than overwriting the snapshot. Custom scans can still load the model,
so this does not guarantee that a 512 MB instance never reaches its limit.

## Background reads

Watchlist performance starts alongside watchlist detail. News and SEC filings
start alongside stock detail. Child cards reuse the parent's request state;
there are no duplicate fetches, cross-account cache changes, or blocking awaits
in the main page. Refresh and mutation paths also refresh performance.

## Deployment checks

Run the isolated PostgreSQL cache test and pre-merge browser journey before
merging. Compare production cold/warm response times and current/peak container
memory after deployment; local tests do not establish a production speedup.
No hosting-plan change or service restart is included.
