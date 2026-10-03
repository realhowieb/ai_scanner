# Admin Research Intelligence Dashboard

## Data inventory

| Metric | Production source | Support |
|---|---|---|
| Users, current plans, signup dates | `users` | Supported |
| Active users | authenticated-session events in `acquisition_events` | Supported for the selected period |
| Paid conversions | `successful_paid_conversion` acquisition events | Supported when the event was recorded |
| Historical tier transitions | no complete transition ledger | Not supported; never inferred from current tier |
| Scan activity | bounded rows from `runs` | Supported |
| Scanner research | `hsf_observations` with `scheduled:*` context | Supported |
| Matured outcomes | `hsf_observation_outcomes` | Supported |
| Signal and cohort | `scanners[]` and `research_cohort` in the immutable observation | Supported |
| HSF Score calibration | explicit persisted HSF score only | Currently partial; Breakout Score is never substituted |
| Stair-Stepper evidence | `day_trader:stair_stepper` observations and outcomes | Supported and isolated |
| System Health | latest `hsf_system_health` snapshot | Supported |
| Autonomous Recovery | latest 100 `hsf_recovery_events` | Supported |

## Statistical rules

- The default evidence threshold is `n >= 30`.
- Every insufficient cell displays its actual sample size.
- Missing returns, MFE and MAE remain null; they are never converted to zero.
- Scheduled scanner research and Stair-Stepper research are separate populations.
- The existing Stair-Stepper verdict supplies the best-supported-window decision and retains its sample and trading-day gates.
- HSF Score calibration is hidden behind an explicit unavailable state when canonical score data is absent.

## Performance and safety

The dashboard is admin-only, read-only and cached for five minutes. Its database
reader executes bounded `SELECT` statements and performs no schema DDL. The
normal application pages do not import or load the analytics bundle. CSV export
contains only the currently filtered research aggregate rows; it contains no
credentials, authentication values or user identities.

The **Observation Explorer** is the row-level companion to these aggregate views.
See `docs/ADMIN_OBSERVATION_EXPLORER.md` for its epoch, integrity, query, and export contracts.
