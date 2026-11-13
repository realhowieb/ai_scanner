# PreBreakout in Admin → Models

The PreBreakout card reads the active database model metadata, latest saved opportunity scores, and training calibration. It does not load a model artifact, rescore a stock, train, or change production behavior.

The target is shown from the active model metadata. Missing metadata stays unknown. PreBreakout is distinct from AI Confidence and ML v4 challengers; it does not inherit their targets or confidence tiers.

Latest rows use prediction-time PreBreakout percentages. Failed/missing captures that carry zero placeholders are unavailable; explicit captured zero remains a valid probability. Model identities are shown per row, and mismatches with the active database model warn about possible fallback or version changes. Superseded Brief observations are excluded.

“View saved inputs” reads only that ticker and the exact displayed scan timestamp. It shows recorded feature values and whether each was defaulted; unknown masks stay unknown. This is input evidence, not feature contribution or a causal explanation. Nothing invokes an LLM automatically.

Training calibration buckets are historical raw-score validation evidence. No live hit-rate curve is fabricated from five-day MFE/MAE: those outcomes cannot establish a future qualifying setup followed by its economic target. A target-compatible forward evaluation is still needed.

Both endpoints require admin access. Queries use uniquely named SELECT expressions and are covered with real PostgreSQL dictionary-row integration. Live acceptance checks admin success and refuses non-admin access.
