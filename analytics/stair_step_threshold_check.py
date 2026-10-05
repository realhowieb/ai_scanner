"""P2-35 step 1 — how the Stair-stepper thresholds behave on a real session.

Replays one session's 1-minute bars: every `step_min` minutes it runs the
production metrics (analytics.stair_step) on the bars up to that moment, the
same as clicking the Day Trader button then, and counts which names would
qualify under a small grid of thresholds around the defaults.

Descriptive only: it counts how often and how steadily names qualify. It never
looks at what happened afterwards (that is the separate, pre-registered outcome
research in analytics.stair_step_research) and changes no defaults, scores or
rankings. Pure functions; the script fetches the bars.
"""
from __future__ import annotations

import datetime as dt
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analytics.stair_step import DEFAULT_WINDOW, is_stair_stepper, stair_step_metrics

DEFAULTS = {"r2_min": 0.80, "max_pullback_pct": 1.0, "min_trend_pct_per_hour": 0.5}
R2_GRID = (0.70, 0.75, 0.80, 0.85, 0.90)
PULLBACK_GRID = (0.5, 1.0, 1.5, 2.0)
TREND_GRID = (0.25, 0.5, 1.0)
STEP_MIN = 5


def _ts(v: Any) -> Optional[dt.datetime]:
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def check_times(open_utc: dt.datetime, close_utc: dt.datetime, window: int,
                step_min: int = STEP_MIN) -> List[dt.datetime]:
    """Moments a user could have clicked: from `window` minutes after the open
    (the first full window) to the close, every `step_min` minutes."""
    out, t = [], open_utc + dt.timedelta(minutes=window)
    while t <= close_utc:
        out.append(t)
        t += dt.timedelta(minutes=step_min)
    return out


def replay_metrics(bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]], times: Sequence[dt.datetime],
                   window: int = DEFAULT_WINDOW) -> Dict[str, List[Dict[str, Any]]]:
    """{symbol: [metrics at each check time]} using only bars up to that time."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym, bars in bars_by_symbol.items():
        timed = sorted(((t, b) for b in bars or [] if (t := _ts(b.get("t"))) is not None), key=lambda x: x[0])
        series, i, seen = [], 0, []
        for t in times:
            while i < len(timed) and timed[i][0] <= t:
                seen.append(timed[i][1])
                i += 1
            series.append(stair_step_metrics(seen[-(window * 3):], window=window))
        out[sym] = series
    return out


def _streaks(flags: Iterable[bool]) -> List[int]:
    runs, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def summarize_cell(metrics: Dict[str, List[Dict[str, Any]]], n_checks: int, *, direction: str,
                   r2_min: float, max_pullback_pct: float, min_trend_pct_per_hour: float,
                   step_min: int = STEP_MIN) -> Dict[str, Any]:
    per_check = [0] * n_checks
    symbols_hit, runs = [], []
    for sym, series in metrics.items():
        flags = [is_stair_stepper(m, r2_min=r2_min, direction=direction, max_pullback_pct=max_pullback_pct,
                                  min_trend_pct_per_hour=min_trend_pct_per_hour) for m in series]
        for k, f in enumerate(flags):
            per_check[k] += int(f)
        if any(flags):
            symbols_hit.append(sym)
        runs.extend(_streaks(flags))
    return {
        "r2_min": r2_min, "max_pullback_pct": max_pullback_pct, "min_trend_pct_per_hour": min_trend_pct_per_hour,
        "is_default": (r2_min, max_pullback_pct, min_trend_pct_per_hour) == tuple(DEFAULTS.values()),
        "avg_hits_per_check": round(mean(per_check), 2) if per_check else 0.0,
        "checks_with_a_hit_pct": round(100.0 * sum(1 for c in per_check if c) / n_checks, 1) if n_checks else 0.0,
        "symbols_hit": len(symbols_hit),
        "median_hold_min": median(runs) * step_min if runs else None,
        "one_check_blips_pct": round(100.0 * sum(1 for r in runs if r == 1) / len(runs), 1) if runs else None,
        "top_symbols": sorted(symbols_hit)[:15],
    }


def data_quality(metrics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    counts = {"ok": 0, "sparse": 0, "insufficient": 0}
    for series in metrics.values():
        for m in series:
            counts[m.get("status", "insufficient")] = counts.get(m.get("status", "insufficient"), 0) + 1
    total = sum(counts.values()) or 1
    thin = sorted(s for s, series in metrics.items()
                  if series and sum(1 for m in series if m.get("status") == "ok") < len(series) / 2)
    return {"checks": total, **{f"{k}_pct": round(100.0 * v / total, 1) for k, v in counts.items()},
            "mostly_thin_symbols": thin}


def build_report(bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]], session: Tuple[dt.datetime, dt.datetime],
                 *, window: int = DEFAULT_WINDOW, step_min: int = STEP_MIN,
                 directions: Sequence[str] = ("up", "down")) -> Dict[str, Any]:
    open_utc, close_utc = session
    times = check_times(open_utc, close_utc, window, step_min)
    metrics = replay_metrics(bars_by_symbol, times, window)
    grids = {}
    for direction in directions:
        grids[direction] = [summarize_cell(metrics, len(times), direction=direction, r2_min=r2,
                                           max_pullback_pct=pb, min_trend_pct_per_hour=tr, step_min=step_min)
                            for r2 in R2_GRID for pb in PULLBACK_GRID for tr in TREND_GRID]
    return {"session_open_utc": open_utc.isoformat(), "session_close_utc": close_utc.isoformat(),
            "window_bars": window, "step_min": step_min, "check_times": len(times),
            "symbols": sorted(bars_by_symbol), "data_quality": data_quality(metrics),
            "defaults": dict(DEFAULTS), "grids": grids}


def render_markdown(report: Dict[str, Any]) -> str:
    q = report["data_quality"]
    lines = [
        "# Stair-stepper threshold check (P2-35)",
        "",
        f"Session {report['session_open_utc'][:10]} · {len(report['symbols'])} symbols · "
        f"{report['window_bars']}-bar window · checked every {report['step_min']} min "
        f"({report['check_times']} checks).",
        "",
        "Counts how often names would qualify if you clicked the Day Trader button at each check. "
        "It does not look at what happened next; that is the separate outcome research.",
        "",
        f"**Data:** {q['ok_pct']}% of checks had enough 1-minute bars, {q['sparse_pct']}% too gappy, "
        f"{q['insufficient_pct']}% too few bars.",
    ]
    if q["mostly_thin_symbols"]:
        lines.append(f"Mostly too thin to judge: {', '.join(q['mostly_thin_symbols'][:20])}.")
    lines += ["", "How to read it: too strict means almost no check finds anything; too loose means many "
              "names, mostly one-check blips. Defaults are marked ★.", ""]
    for direction, cells in report["grids"].items():
        lines += [f"## Direction: {direction}", "",
                  "| R² ≥ | Max pullback % | Min trend %/hr | Avg names per check | Checks with a hit | "
                  "Names that ever qualified | Median hold (min) | One-check blips |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in cells:
            star = " ★" if c["is_default"] else ""
            hold = "—" if c["median_hold_min"] is None else f"{c['median_hold_min']:g}"
            blips = "—" if c["one_check_blips_pct"] is None else f"{c['one_check_blips_pct']}%"
            lines.append(f"| {c['r2_min']:.2f}{star} | {c['max_pullback_pct']:g} | {c['min_trend_pct_per_hour']:g} | "
                         f"{c['avg_hits_per_check']} | {c['checks_with_a_hit_pct']}% | {c['symbols_hit']} | "
                         f"{hold} | {blips} |")
        default = next((c for c in cells if c["is_default"]), None)
        if default and default["top_symbols"]:
            lines += ["", f"Qualified at the defaults: {', '.join(default['top_symbols'])}."]
        lines.append("")
    return "\n".join(lines) + "\n"
