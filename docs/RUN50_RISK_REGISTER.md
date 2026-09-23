# Run 50 — risk register

Severity: CRITICAL / HIGH / MEDIUM / LOW. Release-blocker = P0.

| ID | Category | Description | Severity | Likelihood | Evidence | Mitigation | Blocker? | Action |
|----|----------|-------------|----------|-----------|----------|-----------|----------|--------|
| R1 | Effectiveness | No live evidence HSF selections beat controls | HIGH | Certain now | Run 48 INSUFFICIENT; cohorts began 2026-09-23 | Accumulate 2–4 wks + rerun Run 48; disclose "unvalidated" | P1 (paid) | Schedule maturation; disclosure |
| R2 | Market data | Single provider (Alpaca) dependency | HIGH | Medium | one provider in data/ | Document failover; monitor provider errors | P1 | Failover plan |
| R3 | Product/legal | Performance/edge claims without evidence | HIGH | Medium | product language | Add "educational, not advice, effectiveness unvalidated" | P1 | Disclosure copy |
| R4 | Persistence | perf_history/cached-universe on ephemeral runner FS | MEDIUM | High | artifacts/ on GH Actions | Move perf_history to Neon; universe re-fetches live anyway | P2 | Durable telemetry |
| R5 | Multi-user | Neon path not load-tested at scale | MEDIUM | Medium | no load test | Load-test before paid beta | P1 (paid) | Load test |
| R6 | Observability | ~1,066 broad excepts can hide critical-path failures | MEDIUM | Medium | grep | Narrow excepts on scan/persist/ML paths; rely on Sentry | P2 | Targeted audit |
| R7 | ML | Live predictive value unvalidated; stale-model risk | MEDIUM | Medium | Run 48; model in Neon | Validate via Run 48; stale-model check | P2 | Validate |
| R8 | UX | Streamlit density/mobile limits | MEDIUM | Medium | UX audit | Progressive disclosure; mobile pass | P2 | UX pass |
| R9 | Security | No formal pentest / secret-rotation policy | MEDIUM | Low | audit | Rotate secrets; pentest before public | P1 (public) | Rotate + pentest |
| R10 | Outcomes | Maturation worker not accruing paired outcomes | MEDIUM | High | run48_readiness INSUFFICIENT | Confirm mature-observations schedule runs | P1 (paid) | Verify worker |
| R11 | Alerts | Potential alert spam at scale | LOW | Low | dedup/cooldown exist | Monitor alert volume | P3 | Monitor |
| R12 | Tech debt | Legacy breakout module, dead scoring versions | LOW | Low | inventory | Recommend cleanup later | P3 | Backlog |

No CRITICAL/P0 risks identified.
