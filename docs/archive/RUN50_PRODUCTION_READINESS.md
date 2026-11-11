# Run 50 — production readiness audit & ship decision

Independent, evidence-based audit of the whole repository. **Audit, not rewrite.**
No scoring/ML/signals/ranking changed. Verdicts are split into three questions
that must not be collapsed.

## Executive summary

HSF is a **technically strong, reliable full-market scanning + intelligence
platform** whose **trading effectiveness is not yet validated** (research cohorts
began accumulating 2026-09-23). It is **ready for a private beta** with clear
disclaimers, **not** yet ready to make performance claims or charge on results.

- **Engineering:** CONDITIONAL SHIP (beta-grade; a few P1s, no P0 blockers).
- **Trading-effectiveness evidence:** INSUFFICIENT.
- **Commercial:** PRIVATE BETA READY.
- **HSF PRODUCT READINESS SCORE: 7.8 / 10.**

## Architecture

Streamlit UI (`ui/`, `pages/`) → scan engine (`scan/engine.py` + `scan/breakout`)
→ Alpaca market data (`data/`) over the canonical **US_MARKET** universe
(`data/us_market_universe.py`) → scheduler (`scheduler/cron_runner.py`, GH Actions
cron) → intelligence (`analytics/opportunity_view` + `ui/opportunities`,
`ml_prebreakout`, `scan/ai_confidence`, `analytics/day_trade_intel`) → alerts /
watchlists / Market Brief / Historical Replay → observation capture
(`analytics/observation_capture`, `db/hsf_observations`) → research cohorts +
outcomes → effectiveness framework (`analytics/signal_effectiveness`) →
`analytics/intelligence_view` (canonical presentation). Persistence: **Neon
Postgres** (user data) + SQLite (local/CI) + GH Actions artifacts (ephemeral
telemetry). ML models stored in Neon, downloaded at runtime with fail-safe
guards.

**Debt (non-blocking):** legacy `scan/breakout` module; ~1,066 broad `except`
(intentional best-effort, reported via Sentry `_capture`); ephemeral
`perf_history.jsonl` / cached-universe artifacts; some Run 37–49 docs may drift.

## What HSF does well

1. **Whole-market coverage** — live 98.4% of ~11.8K eligible, HEALTHY, in ~2 min,
   with a measurable funnel and provenance (source=live).
2. **Operational reliability** — health-gated snapshot promotion, overlap lock,
   failure taxonomy, telemetry (Run 45), all verified live.
3. **Point-in-time integrity** — features vs separate outcomes, leakage
   regression-tested (Runs 43/46/47).
4. **Research discipline** — bounded deterministic cohorts + honest INSUFFICIENT
   effectiveness verdict (no premature tuning).
5. **Consistent explainable intelligence** — one canonical `IntelligenceView`,
   provenance-carrying, no composite score (Run 49).

## What remains risky

Single market-data provider (Alpaca) dependency; effectiveness unvalidated
(cannot claim edge); ephemeral operational artifacts; multi-user path not
load-tested; Streamlit UX density/mobile limits; outcome maturation worker not yet
accumulating paired outcomes.

## Live market validation (current — 2026-09-23, run 35819912828)

| Metric | Value |
|---|---:|
| Universe source | **live** |
| Eligible / attempted / priced | 11,826 / 11,825 / **11,633** |
| Skipped | 192 (all FILTERED_BY_POLICY; provider_trouble 0) |
| Coverage / health | **98.37% / HEALTHY** |
| Candidates | 100 |
| Runtime / throughput | 123.1s / 96.1 sym/s |
| Snapshot promoted | **True** |

## Research evidence status

Run 48 verdict stands: **INSUFFICIENT LIVE DATA**. Cohorts began 2026-09-23;
maturation worker not yet producing paired candidate/control matured outcomes.
`intelligence_view.run48_readiness` returns INSUFFICIENT → CONTINUE_ACCUMULATING.
**System functionality ≠ trading effectiveness** — HSF is technically ready while
its predictive effectiveness is unproven.

## Category scorecard (0–10)

| Category | Score | Evidence / biggest risk |
|---|---:|---|
| Market Coverage | 9 | live 98.4%, no 2K cap, verified / — |
| Market Data Reliability | 8 | isolation+fallback / single-provider dependency |
| Scheduled Reliability | 9 | health/overlap/snapshot verified / — |
| Scanner Engine | 8 | deterministic ranked / legacy module |
| Data Integrity | 9 | PIT-safe, provenance, tested / — |
| Research Infrastructure | 8 | bounded deterministic cohorts / — |
| Outcome Integrity | 8 | horizon-gated, direction-aware / maturation not accruing |
| ML Engineering | 7 | fail-safe, versioned / live value unvalidated |
| Intelligence Consistency | 9 | canonical view, shared engine / — |
| Historical Replay | 9 | leakage regression / — |
| Watchlists | 8 | Neon, per-user, tested / — |
| Alerts | 7 | dedup/cooldown, persisted / spam at scale |
| UX | 6 | functional Streamlit / density, mobile |
| Performance | 8 | ~2 min, 96 sym/s (GOOD) / — |
| Observability | 7 | artifacts+Sentry / ephemeral history, swallowed excepts |
| Security | 8 | secrets clean, sanitized errors / no pentest |
| Deployment | 7 | Streamlit+cron+Neon / ephemeral artifacts |
| Persistence | 8 | Neon durable / artifacts ephemeral |
| Multi-User Readiness | 6 | auth/entitlements/billing scaffolding / not load-tested |
| Monetization Readiness | 5 | billing_service exists / effectiveness unvalidated |
| Documentation | 8 | extensive Run docs / some drift |
| Testing | 9 | 1,368 tests, broad / ML live-value gaps |

**Average ≈ 7.8 / 10.**

## Blockers

- **P0 (BLOCKER):** none found.
- **P1 (before paid beta):** validate effectiveness (accumulate + rerun Run 48);
  clear "educational, effectiveness unvalidated, not investment advice"
  disclosure; load-test multi-user Neon path; move `perf_history` to durable
  store; document single-provider failover.
- **P2 (should fix):** UX density/mobile pass; reduce swallowed-exception blind
  spots on critical paths; schedule the maturation worker to accrue outcomes;
  doc drift cleanup.
- **P3 (backlog):** legacy breakout refactor; dead-code sweep; cost dashboards.

## Security / Persistence / Performance / Deployment / UX

- **Security:** `.gitignore` covers `.env`/keys/secrets; only a `.example`
  tracked; 0 hardcoded secret patterns; errors sanitized (Run 30). No exposed
  credential found. Recommend a rotation + pentest before public launch.
- **Persistence:** user-critical data (watchlists, observations, outcomes, runs,
  alerts) is **Neon-durable**; only telemetry artifacts are ephemeral.
- **Performance:** GOOD — full market in ~2 min; slowest stage is price fetch.
- **Deployment:** Streamlit Cloud (app, from `dev`) + GH Actions cron (scans) +
  Neon; models downloaded at runtime. Ephemeral-artifact assumptions documented.
- **UX:** functional and trader-usable; not redesigned. Prioritized recs:
  progressive disclosure, mobile density, clearer stale/empty states.

## Multi-user / monetization readiness

Auth (`auth/`, `db/users`, email verification, password reset), per-user
watchlists/settings, entitlements, and a `billing_service/` scaffold exist →
**SMALL BETA READY** technically; the shared scan backend means per-user cost is
low. Gating issue for paid tiers is **effectiveness validation + explicit
disclosures**, not infrastructure.

## Cost (categories; provider pricing not fabricated)

Shared backend scan cost (Alpaca data + GH Actions minutes + Neon) is **fixed per
scan**, not per user → scales cheaply to 10/100/1,000 users. Per-user cost is
storage + occasional metrics fetches (watchlist). Unknowns: exact Alpaca/Neon/
hosting tiers at scale — mark as TBD.

## Test results / lint

1,368 passed / 9 skipped; ruff `E9,F,I` clean. No production output changed.

## Final verdicts

- **ENGINEERING READINESS: CONDITIONAL SHIP** — reliable and verified; ship to
  private beta; clear the P1s before paid.
- **TRADING-EFFECTIVENESS EVIDENCE: INSUFFICIENT** — no edge claim permitted yet.
- **COMMERCIAL READINESS: PRIVATE BETA READY** — with disclaimers; not paid-ready.

See `RUN50_RISK_REGISTER.md`, `LAUNCH_CHECKLIST.md`, `POST_LAUNCH_MONITORING.md`.
