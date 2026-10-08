# Watchlist Intelligence and alert rules in hsf-api

The API side. The Streamlit watchlist page has its own live view (Run 42, [WATCHLIST_INTELLIGENCE.md](WATCHLIST_INTELLIGENCE.md)),
built from Day Trader quotes; this one is built from the saved market scans so it needs no provider calls.

```
Market Data
    ↓
Scanner / HSF Intelligence        scheduled scans save runs (db.runs); canonical
    ↓                             ranking = ui.results_intelligence.consolidate_scanner_results
hsf-api
    ├── Watchlist Intelligence    GET /v1/watchlists/{id}/intelligence, /changes
    ├── Alert Rules               /v1/alerts/rules (hsf_alert_rules)
    └── Alert Events              GET /v1/alerts/events (alert_events + hsf_alert_rule_events)
              ↓
      Realtime Alert Delivery     billing_service.realtime_alerts loop in hsf-api
              ↓                   (REALTIME_ALERTS_ENABLED=1); rules ride the same loop
       Email / Web / Future Push  in-app = the event row; email via the worker's SMTP sender
```

Nothing here scores, ranks or calls a market-data provider. Everything is read from
the saved full-market scans the Scanner, Stock Intelligence and Today pages use.

## Watchlist Intelligence

`api/watchlist_intel.py`

* **Source.** `market_observation()` turns the latest saved market run and the one
  before it into per-ticker lookups: the canonical ranked setups
  (`api.scans.run_opportunities`, with rank = position), the raw scan rows (price,
  change, RVOL, EMA 9/21 cross), and the previous run's score and rank.
* **Batch.** A watchlist of any size is dictionary lookups on that object, plus two
  queries for alert counts (rules, ticker alerts). No per-symbol query, no provider call.
  Measured in tests: run results load once per run however many symbols; a 153-symbol
  list answers from cache.
* **Cache.** Key `("wl_obs", latest_run_id, previous_run_id)` in the API's `TTLCache`
  (`api.today._cached`), TTL 30 min. A saved run never changes, so the key itself
  changes when a new scan lands (the run list is re-read every 60 s). Underneath,
  `("run", id)` and `("opps", id)` are the existing 60 s caches.
* **Stale.** `stale` / `freshness` use the `/readyz` rule (a scheduled full-market scan
  was missed). Per symbol: `fresh`, `stale`, `missing` (not in the latest scan),
  `unavailable` (scan data couldn't be read).
* **Fallback.** If the scan can't be read, the list still answers (200) with
  `scan_available: false` and every symbol `unavailable`. Alert counts failing reads as
  `null`. A cache failure never touches alert state: the evaluator keeps its own state
  in the database.
* **Redaction.** PreBreakout fields are null below Premium, and the PreBreakout signal
  and setup label are removed, as on every other surface.
* **Not available (always null):** `company_name`, `price_change` (absolute), `rsi`.
  The scans don't produce them; nothing is estimated.

`/changes` runs the canonical opportunity-event engine
(`analytics.opportunity_events.derive_opportunity_events`: new, dropped, rising/falling
by the canonical ±3, status changes, fading, signals added/removed) on the two latest
runs for the list's symbols, adds each symbol's rank move, and lists the user's rule
alerts since the previous scan.

## Alert rules

`api/alert_rules.py` (rules, evaluator, delivery), `db/alert_rules.py` (storage).

| Rule type | Kind | Fires when |
|---|---|---|
| `HSF_SCORE_ABOVE` | level | score ≥ threshold on a new scan (once per cooldown) |
| `HSF_SCORE_BELOW` | level | ranked and score < threshold on a new scan |
| `HSF_SCORE_CROSS_ABOVE` | transition | last seen < threshold ≤ now |
| `HSF_SCORE_CROSS_BELOW` | transition | last seen ≥ threshold > now |
| `PREBREAKOUT_ACTIVE` | transition | PreBreakout signal off → on (Premium) |
| `SETUP_APPEARED` | transition | not ranked → ranked setup (optional `value`: one setup) |
| `RVOL_ABOVE` | level | scan RVOL ≥ threshold |
| `RANK_IMPROVED` | transition | rank up ≥ threshold places since the last scan the rule saw |

A rule watches one `ticker` or every symbol of one `watchlist_id`.

**Crossing semantics.** `hsf_alert_rule_state` keeps, per (rule, ticker), the last value
seen and the observation (`run:<id>`) it came from. 77 → 82 crosses 80 and fires;
83 → 84 doesn't. The first scan a rule sees only sets the baseline for transitions.
A ticker that drops out of the ranking keeps its last known score as the comparison
point; a ticker absent from the scan entirely changes nothing.

**Evaluation lifecycle.** Each worker loop (60 s) during extended hours (4:00-20:00 ET
weekdays, so the database can sleep overnight) calls `worker_pass()`:
enabled rules + owners' plans → skip when none / no scan / scan stale (state untouched,
so nothing is lost) → plan limits → states → pure check → cooldown → one transaction
(insert events, upsert state, stamp rules) → deliver → record delivery.

**Deduplication.**
1. Per (rule, ticker), an observation is evaluated once (`state.observation_id`).
2. `UNIQUE (rule_id, ticker, observation_id)` on events: a retried pass or a second
   process can't insert the same trigger twice (`ON CONFLICT DO NOTHING`).
3. `cooldown_seconds` per rule (default 1 day for level rules, 1 hour for
   transitions; 0 to 7 days) suppresses repeats; counted as deduplicated.

**Delivery.** `delivery` per channel: `in_app` = `delivered` (the event row is the
in-app alert); `email` = `sent`, `failed` (retried on later passes, up to 3 attempts
within 6 hours), or `skipped` (unverified email, alert emails turned off, no address).
The event is written before any delivery, so a failed channel never loses it. Web
push is not implemented yet.

**Entitlements** (server-side, existing policy only):

| Capability | Free | Pro | Premium | Source |
|---|---|---|---|---|
| Watchlists | 50 | 50 | 50 | `api.user_data.MAX_WATCHLISTS` |
| Symbols per watchlist | no limit | no limit | no limit | none defined |
| Symbols per request | 200 | 200 | 200 | `MAX_TICKERS_PER_REQUEST` |
| Active alerts (rules + ticker alerts) | 1 | 5 | 25 | `ALERT_LIMIT_BY_TIER` |
| `PREBREAKOUT_ACTIVE` | no | no | yes | `can_early_breakout` |
| Email channel | no | yes | yes | `can_email_alerts` |

The evaluator re-applies them every pass (newest rules first), so a downgrade takes
effect at the next scan. `GET /v1/me/capabilities` returns the structure.

**Kill switch.** `HSF_ALERT_RULES_ENABLED=0` on hsf-api stops evaluation (rules stay
saved). The loop only runs where `REALTIME_ALERTS_ENABLED=1`.

**Observability.** JSON log lines: `alert_rules_pass` (observation, rules, evaluated,
triggered, deduplicated, delivered, delivery_failed, ms), `watchlist_intelligence`
(symbols, enriched, missing, ms), plus failures by error type only (never messages,
tokens or addresses). In-process counters: `api.alert_rules.metrics()`.

**Not supported (no reliable server-side data yet):** Day Trader and Stair-Stepper
setups (built from live 1-minute bars on request, not persisted per symbol), RSI,
web push.
