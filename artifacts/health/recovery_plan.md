# HSF Recovery Plan

Generated 2026-09-26T08:57:13.138923+00:00 · health 2026-09-26T08:57:13.159598+00:00 (DEGRADED)

- Production autonomy: **OBSERVE_ONLY** (kill switch off (HSF_AUTONOMY_ENABLED != true))
- Circuit breaker: **CLOSED**
- Would execute: nothing · executing: nothing
- Escalations: none

| Incident | Policy | Action | Risk | Attempts | Cooldown | Reason |
|---|---|---|---|---|---|---|
| maturation:MATURATION_CAP_BINDING | WATCH | — | — | —/— | — min | expected or self-clearing; no recovery needed now |
