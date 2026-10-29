# Launch checklist

## Before PRIVATE BETA (target: ready now, after these)
- [x] Full test suite green (1,368) + ruff clean
- [x] Live US_MARKET scan HEALTHY (98.4%, source=live) with snapshot promotion
- [x] Secrets ignored (.env/keys/secrets.toml) — none committed
- [x] User data Neon-durable (watchlists/observations/outcomes/runs/alerts)
- [x] Add explicit disclosure: "educational; not investment advice; predictive
      effectiveness UNVALIDATED" on scanner/brief/alerts
- [ ] Confirm production Neon + Alpaca secrets set in all three stores
      (GitHub Actions and Render; Streamlit Cloud is gone) — see GO_LIVE.md section 2
- [x] Confirm error monitoring (Sentry DSN) live for the API (2026-10-22)
- [x] Verify scheduled scan + maturation workflows enabled on default branch
      (GitHub default branch `main`; workflows active. Latest checked successful
      `main` runs on 2026-10-10 UTC: Scheduled Market Scans
      `38016239184`, Mature Observations `38015503624`.)

## Before PAID BETA
- [ ] Trading-effectiveness: rerun Run 48 once evidence ≥ MODERATE; publish an
      honest effectiveness summary (or explicitly market as "signals, not returns")
- [ ] Load-test multi-user Neon path (concurrent users, watchlist writes)
      Tooling ready: scripts/neon_load_test.py + the manual "Neon Load Test"
      workflow (run against a Neon branch). Schema-setup deadlocks fixed (P1-44).
- [x] Billing/entitlements verified end-to-end (live purchase 2026-10-28; billing embedded in hsf-api-v1)
- [ ] Move perf_history + operational telemetry to durable storage
- [ ] Document + test market-data provider failover
- [x] Backup/restore runbook for Neon: docs/PROVIDER_FAILOVER_AND_RESTORE.md
- [ ] Run the Database restore drill workflow and record its timings (monthly after that)
- [ ] Raise the Neon restore window above 1 day
- [x] Secret rotation policy — docs/SECRET_ROTATION.md (P1-45)

## Before PUBLIC LAUNCH
- [ ] Security pentest + secret rotation
- [ ] Legal review of all product/marketing language, then publish the Terms of Service and
      Privacy Policy (`HSF_LEGAL_PAGES_ENABLED=1`, see GO_LIVE.md section 4)
- [ ] Scale test (1,000+ users) + cost model confirmed with real provider tiers
- [x] Next.js frontend live at app.hsfinest.com
- [ ] SLA / status page / incident runbook

## Backup/recovery (current-stage)
- Neon point-in-time restore: 1 day on the current plan (confirmed in the runbook).
- Observations/outcomes/runs are append-only (recoverable from Neon).
- Restore steps and the monthly drill: docs/PROVIDER_FAILOVER_AND_RESTORE.md.
