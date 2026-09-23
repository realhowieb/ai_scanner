# Launch checklist

## Before PRIVATE BETA (target: ready now, after these)
- [x] Full test suite green (1,368) + ruff clean
- [x] Live US_MARKET scan HEALTHY (98.4%, source=live) with snapshot promotion
- [x] Secrets ignored (.env/keys/secrets.toml) — none committed
- [x] User data Neon-durable (watchlists/observations/outcomes/runs/alerts)
- [ ] Add explicit disclosure: "educational; not investment advice; predictive
      effectiveness UNVALIDATED" on scanner/brief/alerts
- [ ] Confirm production Neon + Alpaca secrets set in all three stores
      (Streamlit Cloud, GitHub Actions, host) — see three-secret-stores memory
- [ ] Confirm error monitoring (Sentry DSN) live for app + cron
- [ ] Verify scheduled scan + maturation workflows enabled on default branch

## Before PAID BETA
- [ ] Trading-effectiveness: rerun Run 48 once evidence ≥ MODERATE; publish an
      honest effectiveness summary (or explicitly market as "signals, not returns")
- [ ] Load-test multi-user Neon path (concurrent users, watchlist writes)
- [ ] Billing/entitlements verified end-to-end (billing_service/)
- [ ] Move perf_history + operational telemetry to durable storage
- [ ] Document + test market-data provider failover
- [ ] Backup/restore runbook for Neon (below)
- [ ] Secret rotation policy

## Before PUBLIC LAUNCH
- [ ] Security pentest + secret rotation
- [ ] Legal review of all product/marketing language
- [ ] Scale test (1,000+ users) + cost model confirmed with real provider tiers
- [ ] Consider frontend (see Run 50 §24 on Next.js timing)
- [ ] SLA / status page / incident runbook

## Backup/recovery (current-stage)
- Neon provides point-in-time restore (managed) — confirm retention setting.
- Observations/outcomes/runs are append-only (recoverable from Neon).
- No app-side backup needed at beta scale; document Neon restore steps.
