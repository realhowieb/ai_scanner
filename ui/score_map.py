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
    ("Quiet (<20)", float("-inf"), 20.0, "#64748b"),
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


def rank_history(history: List[Dict[str, Any]], tickers: List[str], days: int = 5) -> Dict[str, List[Optional[int]]]:
    """Per-ticker rank (1 = top) across the newest `days` snapshots, oldest-first.

    None where the ticker wasn't in that day's top list.
    """
    recent = list(reversed(history[:days]))  # oldest-first
    out: Dict[str, List[Optional[int]]] = {t: [] for t in tickers}
    for day in recent:
        ranks = {t: i + 1 for i, (t, _s) in enumerate(day.get("scores") or [])}
        for t in tickers:
            out[t].append(ranks.get(t))
    return out


def render_score_map(df) -> None:
    """Treemap + breadth strip for a results DataFrame. Never raises."""
    if st is None or df is None or getattr(df, "empty", True):
        return
    if "BreakoutScore" not in df.columns or "Ticker" not in df.columns:
        return
    try:
        with st.expander("🗺️ Score map", expanded=False):
            view = st.radio(
                "View", ["🗺️ Treemap", "🫧 Bubbles"], horizontal=True,
                key="score_map_view", label_visibility="collapsed",
            )
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

            if view == "🫧 Bubbles":
                _render_bubbles(work)
                _render_top_multiples(work)
                _render_rank_bump(work)
                return

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
            _render_top_multiples(work)
            _render_rank_bump(work)
    except Exception:
        pass


def _render_top_multiples(work, top_n: int = 5) -> None:
    """Small multiples: the top picks' 10-day shapes side by side.

    Uses the Spark10D closes captured at scan time — zero extra downloads.
    Same y-scaling per panel (indexed to first close) so shapes compare fairly.
    """
    try:
        if "Spark10D" not in work.columns:
            return
        top = work.head(top_n)
        rows = [
            (str(r["Ticker"]), r["Spark10D"])
            for _, r in top.iterrows()
            if isinstance(r.get("Spark10D"), (list, tuple)) and len(r["Spark10D"]) >= 5
        ]
        if len(rows) < 2:
            return
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=len(rows),
            subplot_titles=[t for t, _ in rows],
            horizontal_spacing=0.03,
        )
        for i, (_t, closes) in enumerate(rows, start=1):
            base = closes[0] or 1.0
            idx = [(c / base - 1) * 100 for c in closes]
            color = "#16a34a" if idx[-1] >= 0 else "#dc2626"
            fig.add_trace(
                go.Scatter(
                    y=idx, mode="lines", line=dict(color=color, width=2),
                    hovertemplate="%{y:+.1f}%<extra></extra>", showlegend=False,
                ),
                row=1, col=i,
            )
            fig.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1,
                          row=1, col=i)
        fig.update_layout(
            height=140, margin=dict(l=0, r=0, t=24, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_annotations(font_size=12)
        st.caption("**Top picks — last 10 days** (indexed to day 1)")
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
    except Exception:
        pass


def _render_rank_bump(work, top_n: int = 8) -> None:
    """Bump chart: how today's top tickers' RANKS moved over recent snapshots."""
    try:
        from ui.alert_preview import load_score_history

        history = load_score_history()
        if len(history) < 3:
            return
        tickers = [str(t) for t in work["Ticker"].head(top_n).tolist()]
        ranks = rank_history(history, tickers, days=5)
        # Need at least two tickers with two known ranks to be worth drawing.
        drawable = {t: r for t, r in ranks.items() if sum(v is not None for v in r) >= 2}
        if len(drawable) < 2:
            return
        import plotly.graph_objects as go

        days = [h["day"] for h in reversed(history[:5])]
        # Fixed hue order by today's rank (identity, never cycled).
        palette = ["#60a5fa", "#16a34a", "#f59e0b", "#e879f9", "#94a3b8",
                   "#38bdf8", "#f472b6", "#a3e635"]
        fig = go.Figure()
        for i, (t, series) in enumerate(drawable.items()):
            fig.add_trace(
                go.Scatter(
                    x=days[: len(series)], y=series, mode="lines+markers", name=t,
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=7),
                    connectgaps=False,
                    hovertemplate=t + " · %{x|%b %d}: rank %{y}<extra></extra>",
                )
            )
        fig.update_layout(
            height=220, margin=dict(l=0, r=0, t=8, b=0),
            legend=dict(orientation="h", y=1.15, font=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                autorange="reversed", title="rank",
                gridcolor="rgba(128,128,128,0.12)", dtick=5,
            ),
        )
        st.caption("**Rank moves — last 5 scan days** (1 = top of the scan)")
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
    except Exception:
        pass


def bubble_color(chg_pct) -> tuple:
    """(fill, border) diverging by day move; neutral gray when unknown."""
    if chg_pct is None or chg_pct != chg_pct:
        return "rgba(100,116,139,0.15)", "rgba(100,116,139,0.6)"
    alpha = min(abs(float(chg_pct)) / 5.0, 1.0)
    if chg_pct >= 0:
        return f"rgba(22,163,74,{0.10 + alpha * 0.25:.2f})", f"rgba(22,163,74,{0.5 + alpha * 0.5:.2f})"
    return f"rgba(220,38,38,{0.10 + alpha * 0.25:.2f})", f"rgba(220,38,38,{0.5 + alpha * 0.5:.2f})"


def build_bubble_html(data: List[Dict[str, Any]], circles, height: int = 470) -> str:
    """Animated bubble field as self-contained HTML/CSS.

    Each bubble drifts on its own gentle loop (duration/delay derived from a
    hash of the ticker, so layout and motion are stable across reruns) and
    pops slightly on hover with a native tooltip. No JS needed — pure CSS.
    """
    by_id = {d["id"]: d for d in data}
    bubbles = []
    for c in circles:
        d = by_id.get(c.ex["id"])
        if not d:
            continue
        fill, border = bubble_color(d.get("chg"))
        # unit-circle coords -> percentages of the square container
        left = (c.x - c.r + 1) / 2 * 100
        top = (1 - (c.y + c.r)) / 2 * 100
        dia = c.r * 100
        chg = d.get("chg")
        chg_s = f"{chg:+.1f}%" if (chg is not None and chg == chg) else ""
        h = abs(hash(d["id"]))
        dur = 4.5 + (h % 40) / 10.0          # 4.5-8.5s
        delay = (h % 23) / 10.0              # 0-2.2s
        amp = 3 + (h % 5)                    # 3-7px drift
        font = max(8, min(20, int(c.r * height * 0.28)))
        label = (
            f"<div class='bl' style='font-size:{font}px'>{d['id']}"
            + (f"<br><span style='font-size:{max(7, int(font * 0.75))}px'>{chg_s}</span>" if dia > 9 else "")
            + "</div>"
            if dia > 6.5
            else ""
        )
        title = f"{d['id']} · score {d['datum']:.1f}" + (f" · {chg_s} today" if chg_s else "")
        bubbles.append(
            f"<div class='bub' title='{title}' style='left:{left:.2f}%;top:{top:.2f}%;"
            f"width:{dia:.2f}%;height:{dia:.2f}%;background:radial-gradient(circle at 35% 30%, rgba(255,255,255,0.10), {fill});"
            f"border:2px solid {border};animation-duration:{dur:.1f}s;animation-delay:-{delay:.1f}s;"
            f"--amp:{amp}px'>{label}</div>"
        )
    return f"""
<style>
  .bfield {{ position:relative; width:min(100%, {height}px); height:{height}px;
             margin:0 auto; font-family:-apple-system, 'Segoe UI', Roboto, sans-serif; }}
  .bub {{ position:absolute; border-radius:50%; display:flex; align-items:center;
          justify-content:center; text-align:center; color:#e2e8f0; cursor:default;
          animation-name:drift; animation-timing-function:ease-in-out;
          animation-iteration-count:infinite; animation-direction:alternate;
          transition:transform .18s ease; box-sizing:border-box; }}
  .bub:hover {{ transform:scale(1.09); z-index:5; }}
  .bl {{ font-weight:700; line-height:1.15; text-shadow:0 1px 3px rgba(0,0,0,0.6); }}
  @keyframes drift {{ from {{ margin-top:calc(-1 * var(--amp)); }}
                      to   {{ margin-top:var(--amp); }} }}
  @media (prefers-reduced-motion: reduce) {{ .bub {{ animation:none; }} }}
</style>
<div class='bfield'>{"".join(bubbles)}</div>
"""


def _render_bubbles(work, max_bubbles: int = 50) -> None:
    """Crypto-bubbles style packed circles: size = BreakoutScore, color = day %.

    Size carries the setup's strength (magnitude -> area), the green/red ring
    carries today's direction (polarity), and the ticker + % are printed in
    text ink so color is never the only encoding. Animated: each bubble drifts
    gently (CSS only; honors prefers-reduced-motion).
    """
    try:
        import circlify
        import streamlit.components.v1 as components

        top = work.head(max_bubbles)
        data = []
        for _, r in top.iterrows():
            score = float(r["BreakoutScore"])
            if score != score:  # NaN — genuinely unscored
                continue
            data.append({"id": str(r["Ticker"]), "datum": score, "chg": r.get("PctChange")})
        if len(data) < 3:
            st.caption("Not enough scored results for bubbles.")
            return
        # circlify needs positive areas, but BreakoutScore can be negative (weak
        # setups). Shift the whole field so the lowest score maps to a small but
        # visible floor and the highest stays biggest — ordering and relative
        # gaps are preserved; the true score still shows in the label/tooltip.
        lo = min(d["datum"] for d in data)
        hi = max(d["datum"] for d in data)
        floor = max((hi - lo) * 0.12, 1.0)
        circles = circlify.circlify(
            [{"id": d["id"], "datum": d["datum"] - lo + floor} for d in data],
            show_enclosure=False,
        )
        components.html(build_bubble_html(data, circles), height=480)
        st.caption(
            "Bubble size = BreakoutScore (bigger = stronger setup) · "
            "color = today's move. Hover a bubble for details."
        )
    except ImportError:
        st.caption("Bubbles view needs the `circlify` package (redeploying adds it).")
    except Exception:
        pass

