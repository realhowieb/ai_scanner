# HSF Autonomy — Operator Card

**1. Is HSF running?**
Check Admin → 🩺 System Health: the Scanner card should say HEALTHY (18/18 slots),
and "Next expected scan" gives the time. Weekends and holidays are idle by design.

**2. Where do I check health?**
Admin → 🩺 System Health, or the latest **System Health** workflow summary. It
runs weekdays at 17:05 and 23:45 UTC.

**3. HEALTHY:** nothing to do.

**4. DEGRADED:** something is off but not blocking. Watch it; HSF may fix it
itself.

**5. ACTION_REQUIRED:** read `human_escalation.md` (Autonomous Recovery run
summary) and act.

**6. What HSF recovers automatically** (only when enabled; at most 2 tries each,
always verified):
- stale health, readiness or parity-audit reports;
- failed or stale outcome maturation, and provider rate-limit backlog (trading
  days only);
- the universe probe and refresh.

Scans are **never** re-run automatically.

**7. When HSF escalates:**
- database problems, a bad universe, a missed or failed scan, research-capture
  problems;
- 2 failed attempts, the circuit breaker opening;
- a forward-parity or data-quality block;
- forward evidence ready for formal evaluation (needs your approval).

**8. Stop autonomous recovery:** GitHub → Settings → Actions → Variables → set
`HSF_AUTONOMY_ENABLED=false`. Monitoring continues.
(On = `HSF_AUTONOMY_ENABLED=true` **and** `HSF_AUTONOMY_MODE=recover`.)

**9. Restart after the circuit breaker:** fix the cause, confirm a clean health
run, then run
`python -m scripts.autonomous_recovery --reset-circuit --by <name>`.

**10. Don't touch during the forward experiment** (or you void the evidence):
- scoring, weights, ranking, tiers, thresholds, conflict penalties;
- candidate / near-miss / control rules, universe rules, PreBreakout;
- outcome formulas;
- the Run 56 epoch or gates, the Run 58 parity rules;
- stored observations and outcomes.
