# HSF Intelligence API contract (Run 49)

A **contract**, not an API migration. Stable, versioned, null-safe serialization
of the canonical `IntelligenceView` (`analytics/intelligence_view.py`,
`to_dict()`), so a future API / Next.js frontend can render HSF intelligence
**without reimplementing or changing product semantics**. Do NOT build Next.js in
Run 49.

## Schema: `hsf-intelligence-view-1.0`

```jsonc
{
  "schema_version": "hsf-intelligence-view-1.0",
  "symbol": "NVDA",
  "timestamp": "2026-09-22T14:00:00+00:00",

  // 1. WHAT IS HAPPENING
  "direction": "LONG",              // LONG|SHORT|NEUTRAL|MIXED|null
  "lifecycle": "STRENGTHENING",     // NEW|ACTIVE|STRENGTHENING|WEAKENING|RESOLVED
  "tier": null,                     // existing status if present, else null (UNKNOWN)

  // 2. HOW IMPORTANT — scores side-by-side, NEVER combined
  "scores": {
    "opportunity_score": 87,        // 0–100 | null
    "prebreakout": 0.74,            // 0–1 or 0–100 | null
    "ml_probability": 0.68,         // 0–1 | null
    "alert_priority": "HIGH"        // HIGH|MEDIUM|LOW  (attention, not prediction)
  },
  "score_metadata": { /* label/range/higher_means/source/version per score */ },

  // 3. WHY / 4. WHAT RISK — every factor traces to a real value
  "agreement": { "count": 3, "scanners": ["PreBreakout","Momentum","Unusual Volume"] },
  "confirmations": [ { "code":"RVOL_EXPANSION","label":"RVOL 3.1x",
                       "source":"indicators.rvol","value":3.1,"severity":"info" } ],
  "conflicts":     [ { "code":"EXTENDED_ABOVE_VWAP","label":"Extended above VWAP",
                       "source":"indicators.vs_vwap_pct","value":4.2,"severity":"warn" } ],
  "supporting_factors": [ /* == confirmations */ ],
  "caution_factors":    [ /* == conflicts */ ],
  "primary_setup": "PreBreakout",
  "changes_since_prior": [ "RVOL: 1.8x → 3.1x", "Moved above VWAP" ],
  "next_confirmation": "Trend strength (ADX ≥ 20)",   // or null

  // 5. OTHER INTELLIGENCE — via scores above (prebreakout / ml_probability)

  // 6. HOW FRESH / TRUSTWORTHY (scan-level, NOT per-stock prediction confidence)
  "scan_health": { "coverage_health": "HEALTHY",
                   "note": "Describes the SCAN's market coverage — not the reliability of this stock's signal." },
  "research_evidence": { "level": "UNVALIDATED", "note": "no production recommendation" },
  "freshness": "Fresh",             // Fresh|Partial Data|Stale
  "staleness": { "intelligence_as_of": "...", "market_data_as_of": "...", "scan_run_id": "run1" },

  "provenance": { "schema":"hsf-obs-1.0", "hsf_score":"1.0",
                  "prebreakout_model":"prebreakout-xgb-v16",
                  "ai_confidence_model":"ai-confidence-xgb-v1", "dt_score":"run32-v1" }
}
```

## Field rules

- **Null-safe:** any unavailable value is `null` (render as "Unknown"/
  "Unavailable"), never 0/False/Neutral.
- **Deterministic:** identical inputs → byte-identical `to_dict()` (keys sorted;
  `schema_version` first).
- **No composite score:** the four `scores` are independent and must be shown
  separately; never sum/blend them into an "HSF Score".
- **Scan health ≠ confidence:** `scan_health` is scan-level coverage; do not label
  it as the stock's prediction confidence.
- **Research evidence:** only `UNVALIDATED/PRELIMINARY/MODERATE/STRONG`; fixture/
  synthetic results must never be surfaced as live evidence (default UNVALIDATED).
- **Point-in-time:** for Historical Replay, build the view from the historical
  observation; never recompute against current data.

## Progressive disclosure (UI guidance)

- **Table row:** symbol, direction, lifecycle, opportunity_score, alert_priority.
- **Detail:** confirmations, conflicts, other scores (prebreakout/ml),
  next_confirmation, scan_health, research_evidence, provenance, staleness.

Do not render every field on every screen (Run 49 §23).

## Frontend migration readiness

A future API returns `IntelligenceView.to_dict()` per symbol; the frontend renders
the six-section hierarchy above. Because the view is a pure adapter over existing
outputs, the frontend inherits HSF's exact trading/ranking semantics with no
reimplementation. Versioning is via `schema_version`; additive fields bump the
minor, breaking shape bumps the major.
