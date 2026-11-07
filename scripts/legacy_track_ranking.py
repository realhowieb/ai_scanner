"""Frozen audit-only selection from production 1fb35f5e; never used for new performance."""
from math import isfinite
from typing import List

from analytics.track_record import _symbol_column


def legacy_ranked_symbols(df, ranking: str, top_n: int) -> List[str]:
    """Top-N symbols under a ranking method ('breakout' | 'prebreakout')."""
    sym_col = _symbol_column(df)
    if not sym_col:
        return []
    work = df
    if ranking == "prebreakout":
        from analytics.prediction_provenance import models_from_row

        # Use only prediction-time evidence. Loading today's model here rewrites
        # past selections whenever the champion or its calibration changes.
        def saved_probability(row):
            evidence = models_from_row(row).get("prebreakout")
            if isinstance(evidence, dict):
                if evidence.get("status") != "captured":
                    return None
                value, scale = evidence.get("calibrated_probability"), 1.0
            elif "PreBreakoutProb%" in row:
                value, scale = row.get("PreBreakoutProb%"), 100.0
            else:
                value, scale = row.get("PreBreakoutProb"), 1.0
            try:
                value = float(value) / scale
                return value if isfinite(value) and 0.0 <= value <= 1.0 else None
            except (TypeError, ValueError):
                return None

        probabilities = df.apply(saved_probability, axis=1)
        # Legacy unavailable-model fallbacks wrote all zeros; never infer an order
        # from those, or from snapshots with no saved prediction.
        valid = probabilities.notna()
        if not valid.any() or probabilities[valid].max() <= 0:
            return []
        work = df.loc[valid].assign(_saved_probability=probabilities[valid]).sort_values(
            "_saved_probability", ascending=False, kind="stable")
    else:  # breakout
        if "BreakoutScore" in df.columns:
            work = df.sort_values("BreakoutScore", ascending=False)
    return [str(s).upper() for s in work[sym_col].head(top_n).tolist() if str(s).strip()]
