"""Breakout Score Map: Finviz-style treemap of the current scan results.

Tile size = 20-day dollar volume (how much the name matters / how tradable),
tile color = BreakoutScore (sequential single hue — magnitude, not polarity),
label = ticker + score. A breadth bar above it buckets the results so the
day's overall temperature reads at a glance. Pure helpers are unit-testable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import streamlit as st
except Exception:  # pragma: no cover - headless envs
    pd = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

# Score buckets for the breadth strip (clipped ceiling is ~145).
BUCKETS = (
    ("Quiet (<20)", 0.0, 20.0, "#64748b"),
    ("Warm (20-40)", 20.0, 40.0, "#60a5fa"),
    ("Hot (40+)", 40.0, float("inf"), "#16a34a"),
)


def bucket_counts(scores: List[float]) -> Dict[str, int]:
    """Count scores per breadth bucket."""
    out = {name: 0 for name, *_ in BUCKETS}
    for s in scores:
        try:
            val = float(s)
        except (TypeError, ValueError):
            continue
        for name, lo, hi, _color in BUCKETS:
            if lo <= val < hi:
                out[name] += 1
                break
    return out


def render_score_map(df) -> None:
    """Treemap + breadth strip for a results DataFrame. Never raises."""
    if st is None or df is None or getattr(df, "empty", True):
        return
    if "BreakoutScore" not in df.columns or "Ticker" not in df.columns:
        return
    try:
        with st.expander("🗺️ Score map", expanded=False):
            work = df.copy()
            work["BreakoutScore"] = pd.to_numeric(work["BreakoutScore"], errors="coerce")
            work = work.dropna(subset=["BreakoutScore"]).head(60)
            if work.empty:
                st.caption("No scored results to map.")
                return

            # Breadth strip: the day's temperature in one line.
            counts = bucket_counts(work["BreakoutScore"].tolist())
            total = sum(counts.values()) or 1
            parts = [
                f"<span style='color:{color}'>■</span> {name}: <b>{counts[name]}</b>"
                f" ({counts[name] / total:.0%})"
                for name, *_rest, color in [(b[0], b[1], b[2], b[3]) for b in BUCKETS]
            ]
            st.markdown(
                "<div style='font-size:13px'>" + " &nbsp; ".join(parts) + "</div>",
                unsafe_allow_html=True,
            )

            # Tile size: dollar volume when present (importance), else equal.
            if "DollarVol20" in work.columns:
                size = pd.to_numeric(work["DollarVol20"], errors="coerce").fillna(0.0)
                size = size.clip(lower=size[size > 0].min() if (size > 0).any() else 1.0)
            else:
                size = pd.Series(1.0, index=work.index)

            import plotly.graph_objects as go

            chg = (
                pd.to_numeric(work.get("PctChange"), errors="coerce")
                if "PctChange" in work.columns
                else pd.Series(float("nan"), index=work.index)
            )
            custom = [
                (s, c if c == c else None)
                for s, c in zip(work["BreakoutScore"], chg)
            ]
            fig = go.Figure(
                go.Treemap(
                    labels=[
                        f"{t}<br>{s:.0f}"
                        for t, s in zip(work["Ticker"], work["BreakoutScore"])
                    ],
                    parents=[""] * len(work),
                    values=size.tolist(),
                    marker=dict(
                        colors=work["BreakoutScore"].tolist(),
                        colorscale=[[0, "#1e293b"], [0.5, "#2563eb"], [1, "#60a5fa"]],
                        cmin=0,
                        cmax=max(60.0, float(work["BreakoutScore"].max())),
                        colorbar=dict(title="Score", thickness=10),
                        line=dict(width=2, color="rgba(0,0,0,0.35)"),
                    ),
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{label}</b><br>size = $ volume (20d)"
                        "<br>score %{customdata[0]:.1f}<extra></extra>"
                    ),
                    textfont=dict(size=13),
                )
            )
            fig.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
            st.caption(
                "Tile size = 20-day dollar volume (liquidity) · color = BreakoutScore. "
                "Big bright tiles = strong setups in tradable names."
            )
    except Exception:
        pass
