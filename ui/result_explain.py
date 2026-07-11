"""Plain-English "why this passed" explanations for scan results.

Turns the numeric columns a row already carries into one short human line,
e.g. "2.4× avg volume · +4.2% gap · at 20d high · earnings in 3d ⚠️".
Pure formatting — no data fetches — so it is safe on every render path and
testable without pandas (explain_row takes any Mapping).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

WHY_COL = "Why"
_MAX_REASONS = 4


def _num(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    """First present, numeric, non-NaN value among keys."""
    for key in keys:
        if key not in row:
            continue
        val = row[key]
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN
            continue
        return f
    return None


def explain_row(row: Mapping[str, Any]) -> str:
    """One short line of the strongest reasons this row passed the scan."""
    reasons: list[str] = []

    rel_vol = _num(row, "VolRel20", "RelVol")
    if rel_vol is not None and rel_vol >= 1.5:
        reasons.append(f"{rel_vol:.1f}× avg volume")

    gap = _num(row, "GapPct")
    if gap is not None and abs(gap) >= 2.0:
        reasons.append(f"{gap:+.1f}% gap")

    trend10 = _num(row, "Trend10D%")
    trend20 = _num(row, "Trend20D%")
    if trend10 is not None and abs(trend10) >= 5.0:
        reasons.append(f"{trend10:+.0f}% over 10d")
    elif trend20 is not None and abs(trend20) >= 8.0:
        reasons.append(f"{trend20:+.0f}% over 20d")

    pos20 = _num(row, "BreakoutPos20D")
    if pos20 is not None and pos20 >= 0.97:
        reasons.append("at 20d high")

    rs = _num(row, "RSvsSPY")
    if rs is not None and rs >= 5.0:
        reasons.append("outpacing SPY")

    prob = _num(row, "PreBreakoutProb%")
    if prob is not None and prob >= 70.0:
        reasons.append(f"{prob:.0f}% model conf")

    # Risk flag rather than a passing reason — always keep it if present.
    earn_days = _num(row, "📅 Earnings in X days", "earnings_in_days")
    earn_flag = (
        f"earnings in {earn_days:.0f}d ⚠️"
        if earn_days is not None and 0 <= earn_days <= 5
        else None
    )

    out = reasons[: _MAX_REASONS - (1 if earn_flag else 0)]
    if earn_flag:
        out.append(earn_flag)
    if not out:
        score = _num(row, "BreakoutScore")
        return f"score {score:g}" if score is not None else ""
    return " · ".join(out)


def add_why_column(df):
    """Insert a Why column (after Ticker when present). No-op on empty/None."""
    if df is None or getattr(df, "empty", True) or WHY_COL in getattr(df, "columns", ()):
        return df
    df = df.copy()
    why = [explain_row(row) for row in df.to_dict(orient="records")]
    insert_at = 1 if "Ticker" in df.columns else 0
    df.insert(min(insert_at, len(df.columns)), WHY_COL, why)
    return df
