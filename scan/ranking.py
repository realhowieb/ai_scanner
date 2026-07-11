"""Default result ranking: PreBreakout model first, BreakoutScore fallback.

The daily track-record A/B (signal_track_record) measured PreBreakout-ranked
picks outperforming BreakoutScore-ranked picks (5d excess vs SPY), so results
are presented model-first when the model actually scored. All-zero
probabilities mean no model loaded — never a ranking — so we fall back to
BreakoutScore rather than presenting stored order as a model opinion.
"""
from __future__ import annotations

PREBREAKOUT_COL = "PreBreakoutProb%"
FALLBACK_COL = "BreakoutScore"


def apply_default_ranking(df):
    """Return df ordered by the best available signal. Never raises."""
    try:
        if df is None or getattr(df, "empty", True):
            return df
        if PREBREAKOUT_COL in df.columns and float(df[PREBREAKOUT_COL].max() or 0.0) > 0.0:
            return df.sort_values(PREBREAKOUT_COL, ascending=False).reset_index(drop=True)
        if FALLBACK_COL in df.columns:
            return df.sort_values(FALLBACK_COL, ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return df
