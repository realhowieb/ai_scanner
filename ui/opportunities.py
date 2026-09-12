"""HSF Opportunity Score — rank the brief's signals into Top Opportunities.

Pure, testable logic (no Streamlit): aggregates the market brief's existing
signal lists per ticker, computes a 0-100 confluence-weighted score from ONLY
data that exists, and builds a deterministic plain-English explanation. Never
fabricates RVOL, EMA distance, resistance, earnings, or confidence numbers.

HSF OPPORTUNITY SCORE (0-100) — confluence-weighted, real-data-only
-------------------------------------------------------------------
The app's thesis is *confluence* (a name showing up across independent
scanners), so breadth of confirming signals is the heaviest component. Each
component contributes only when its underlying data is present; a missing input
scores 0 and never invents a value.

  signals_component (0-48): breadth of independent confirming signals.
      Each of {breakout, golden_cross, prebreakout, gapper, gainer} = 12 pts,
      capped at 48 (so ~4 confirming signals maxes it).
  model_component  (0-38): the strongest available model score, whichever
      exists — normalized BreakoutScore (score/60 * 38, capped) OR the
      PreBreakout model probability (prob/100 * 38). Not additive: we take the
      max so two model views don't double-count.
  momentum_component (0-14): positive intraday move only —
      clamp(chg_pct, 0, 6) / 6 * 14. Negative moves add nothing here.
  fading_penalty (-20): applied when the name also shows a 'loser'/fading tag
      (gap fade / reversal risk), so a confirmed-but-fading name is demoted.

  raw   = signals + model + momentum - fading_penalty
  score = round(clamp(raw, 0, 100))

Status tiers: STRONG (>=75 and not fading) · CAUTION (fading, or <50) · WATCH.

Example numbers in the product spec are illustrative — actual scores depend on
each session's real signal data.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

# Component weights (documented above). Tune here; keep the docstring in sync.
_SIGNAL_POINTS = 12
_SIGNAL_CAP = 48
_MODEL_CAP = 38
_BREAKOUT_SCORE_FULL = 60.0  # BreakoutScore value treated as "full marks"
_MOMENTUM_CAP = 14
_MOMENTUM_FULL_PCT = 6.0
_FADING_PENALTY = 20

# Positive confirming signals (order = display priority for "primary setup").
_POSITIVE_SIGNALS = ["golden_cross", "breakout", "prebreakout", "gapper", "gainer"]
_SETUP_LABELS = {
    "golden_cross": "Golden Cross",
    "breakout": "Breakout",
    "prebreakout": "PreBreakout",
    "gapper": "Gapper",
    "gainer": "Momentum",
    "fading": "Gap Fading",
}

_FLAG_RE = re.compile(r"[⚠🚩].*$")


def _default_base_ticker(raw: Any) -> str:
    """Strip earnings/warning flag suffixes and normalize to a bare symbol."""
    if raw is None:
        return ""
    t = _FLAG_RE.sub("", str(raw)).strip().upper()
    return t.split()[0] if t else ""


def build_opportunities(
    data: Dict[str, Any],
    *,
    base_ticker: Optional[Callable[[Any], str]] = None,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Aggregate the brief's signal lists into ranked opportunities.

    Returns a list of dicts (highest score first), each carrying the score, the
    primary setup, confirming-signal count, status, and the real underlying
    fields used (for the deterministic explanation). Safe on empty/partial data.
    """
    norm = base_ticker or _default_base_ticker
    if not isinstance(data, dict):
        return []

    acc: Dict[str, Dict[str, Any]] = {}

    def slot(raw: Any) -> Optional[Dict[str, Any]]:
        t = norm(raw)
        if not t:
            return None
        return acc.setdefault(
            t,
            {"ticker": t, "signals": set(), "breakout_score": None,
             "prob": None, "chg_pct": None, "gap_pct": None},
        )

    for g in (data.get("gappers") or []):
        s = slot(g.get("ticker"))
        if s is None:
            continue
        s["signals"].add("gapper")
        if g.get("gap_pct") is not None:
            s["gap_pct"] = _to_float(g.get("gap_pct"))
        if g.get("chg_pct") is not None:
            s["chg_pct"] = _to_float(g.get("chg_pct"))

    for t in (data.get("golden") or []):
        s = slot(t)
        if s is not None:
            s["signals"].add("golden_cross")

    for pair in (data.get("top_setups") or []):
        try:
            t, score = pair
        except (TypeError, ValueError):
            continue
        s = slot(t)
        if s is not None:
            s["signals"].add("breakout")
            s["breakout_score"] = _to_float(score)

    for p in (data.get("picks") or []):
        s = slot(p.get("symbol"))
        if s is not None:
            s["signals"].add("prebreakout")
            s["prob"] = _to_float(p.get("prob"))

    for pair in (data.get("gainers") or []):
        try:
            t, chg = pair
        except (TypeError, ValueError):
            continue
        s = slot(t)
        if s is not None:
            s["signals"].add("gainer")
            if s["chg_pct"] is None:
                s["chg_pct"] = _to_float(chg)

    for pair in (data.get("losers") or []):
        try:
            t, chg = pair
        except (TypeError, ValueError):
            continue
        s = slot(t)
        if s is not None:
            s["signals"].add("fading")
            if s["chg_pct"] is None:
                s["chg_pct"] = _to_float(chg)

    opportunities = []
    for s in acc.values():
        pos = [sig for sig in _POSITIVE_SIGNALS if sig in s["signals"]]
        fading = "fading" in s["signals"]
        # A pure single-list name with no confluence and no model score isn't an
        # "opportunity" — require either confluence (2+) or a model score.
        if len(pos) < 2 and s["breakout_score"] is None and s["prob"] is None:
            continue
        bd = score_breakdown(
            n_signals=len(pos),
            breakout_score=s["breakout_score"],
            prob=s["prob"],
            chg_pct=s["chg_pct"],
            fading=fading,
        )
        score = bd["score"]
        opportunities.append({
            "ticker": s["ticker"],
            "score": score,
            "score_version": HSF_SCORE_VERSION,
            "score_components": {
                "signals_component": bd["signals_component"],
                "model_component": bd["model_component"],
                "momentum_component": bd["momentum_component"],
                "fading_penalty": bd["fading_penalty"],
            },
            "primary_setup": _primary_setup(pos, fading),
            "n_signals": len(pos),
            "signals": pos,
            "fading": fading,
            "status": _status(score, fading),
            "breakout_score": s["breakout_score"],
            "prob": s["prob"],
            "chg_pct": s["chg_pct"],
            "gap_pct": s["gap_pct"],
        })

    opportunities.sort(key=lambda o: o["score"], reverse=True)
    return opportunities[: max(0, int(top_n))]


# HSF score version — bump when the formula/weights change. Historical frozen
# opportunities keep the version they were scored under, so calibration can
# compare v1.0 vs a future v1.1 without overwriting history.
HSF_SCORE_VERSION = "1.0"


def score_breakdown(
    *,
    n_signals: int,
    breakout_score: Optional[float],
    prob: Optional[float],
    chg_pct: Optional[float],
    fading: bool,
) -> Dict[str, Any]:
    """The 0-100 HSF score AND its components (for freezing/calibration).

    Missing inputs contribute 0 (never a fabricated value). Returns the four
    documented components plus the final clamped score.
    """
    signals_component = float(min(int(n_signals) * _SIGNAL_POINTS, _SIGNAL_CAP))

    model_candidates = [0.0]
    if breakout_score is not None:
        model_candidates.append(min(float(breakout_score) / _BREAKOUT_SCORE_FULL, 1.0) * _MODEL_CAP)
    if prob is not None:
        model_candidates.append(min(max(float(prob), 0.0) / 100.0, 1.0) * _MODEL_CAP)
    model_component = max(model_candidates)

    momentum_component = 0.0
    if chg_pct is not None and chg_pct > 0:
        momentum_component = min(float(chg_pct), _MOMENTUM_FULL_PCT) / _MOMENTUM_FULL_PCT * _MOMENTUM_CAP

    fading_penalty = float(_FADING_PENALTY) if fading else 0.0
    raw = signals_component + model_component + momentum_component - fading_penalty
    score = int(round(max(0.0, min(100.0, raw))))
    return {
        "signals_component": round(signals_component, 2),
        "model_component": round(model_component, 2),
        "momentum_component": round(momentum_component, 2),
        "fading_penalty": round(fading_penalty, 2),
        "score": score,
    }


def build_opportunity_score(
    *,
    n_signals: int,
    breakout_score: Optional[float],
    prob: Optional[float],
    chg_pct: Optional[float],
    fading: bool,
) -> int:
    """Compute the 0-100 HSF Opportunity Score. See module docstring for the
    formula. Missing inputs contribute 0 (never a fabricated value)."""
    return score_breakdown(
        n_signals=n_signals, breakout_score=breakout_score, prob=prob,
        chg_pct=chg_pct, fading=fading,
    )["score"]


def _primary_setup(pos_signals: List[str], fading: bool) -> str:
    for sig in _POSITIVE_SIGNALS:
        if sig in pos_signals:
            return _SETUP_LABELS[sig]
    return _SETUP_LABELS["fading"] if fading else "Signal"


def _status(score: int, fading: bool) -> str:
    if fading:
        return "CAUTION"
    if score >= 75:
        return "STRONG"
    if score < 50:
        return "CAUTION"
    return "WATCH"


def build_opportunity_explanation(
    opp: Dict[str, Any],
    *,
    earnings_today: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Deterministic 'why it ranked' + risk flags from real fields only.

    Returns {"reasons": [...], "risks": [...]}. No invented metrics: every line
    is backed by a value present on the opportunity or the earnings list.
    """
    reasons: List[str] = []
    risks: List[str] = []
    signals = opp.get("signals") or []

    if "golden_cross" in signals:
        reasons.append("Fresh EMA 9/21 golden cross")
    if "breakout" in signals:
        bs = opp.get("breakout_score")
        reasons.append(f"Breakout setup (score {bs:g})" if bs is not None else "Breakout setup")
    if "prebreakout" in signals:
        prob = opp.get("prob")
        reasons.append(f"PreBreakout model confidence {prob:g}%" if prob is not None else "PreBreakout model pick")
    if "gapper" in signals:
        gap = opp.get("gap_pct")
        reasons.append(f"Gapping {gap:+.1f}%" if gap is not None else "Gapping today")
    if "gainer" in signals:
        chg = opp.get("chg_pct")
        reasons.append(f"Positive momentum {chg:+.1f}%" if chg is not None else "Positive momentum")

    n = int(opp.get("n_signals") or 0)
    if n >= 2:
        reasons.append(f"Confluence — appears across {n} independent signals")

    if opp.get("fading"):
        chg = opp.get("chg_pct")
        risks.append(f"Fading / reversal risk ({chg:+.1f}%)" if chg is not None else "Fading / reversal risk")
    et = {str(x).upper() for x in (earnings_today or [])}
    if opp.get("ticker", "").upper() in et:
        risks.append("Earnings reporting today")

    return {"reasons": reasons, "risks": risks}


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Run 18A — market regime, snapshot comparison, change summary, watch-next.
# All deterministic and real-data-only (Claude may *explain* a regime later, but
# never determines it). Every function is safe on missing/partial inputs.
# ---------------------------------------------------------------------------

_STATUS_RANK = {"CAUTION": 1, "WATCH": 2, "STRONG": 3}


def classify_market_regime(
    *,
    spy_chg: Optional[float],
    qqq_chg: Optional[float],
    breadth: Optional[tuple],
    sectors: Optional[List[tuple]] = None,
) -> Dict[str, Any]:
    """Deterministic market regime from data the app already has.

    Regimes: TRENDING BULLISH · RISK-ON · MIXED / CHOPPY · RISK-OFF ·
    TRENDING BEARISH · HIGH VOLATILITY. Never invents VIX/volatility — the
    'HIGH VOLATILITY' label is only used for a *real* index/breadth divergence
    (a large index move contradicted by breadth). Returns {regime, interpretation};
    regime is None when there is no index data to classify from.

    Method (transparent):
      idx        = mean of available SPY/QQQ % changes.
      breadth_r  = advancers / (advancers+decliners), when breadth exists.
      sec_pos    = fraction of sectors positive, when sectors exist.
      broad_up   = breadth_r >= 0.60 OR sec_pos >= 0.66
      broad_dn   = breadth_r <= 0.40 OR sec_pos <= 0.34
      TRENDING BULLISH : idx >= 0.5 and broad_up
      TRENDING BEARISH : idx <= -0.5 and broad_dn
      HIGH VOLATILITY  : |idx| >= 1.0 and breadth clearly disagrees with idx sign
      RISK-ON          : idx >= 0.15 and not broad_dn
      RISK-OFF         : idx <= -0.15 and not broad_up
      else             : MIXED / CHOPPY
    """
    idx_vals = [v for v in (spy_chg, qqq_chg) if v is not None]
    if not idx_vals:
        return {"regime": None, "interpretation": None}
    idx = sum(idx_vals) / len(idx_vals)

    breadth_r = None
    if breadth:
        adv, dec = (breadth[0] or 0), (breadth[1] or 0)
        total = adv + dec
        if total > 0:
            breadth_r = adv / total

    sec_pos = None
    sec_list = [c for (_n, c) in (sectors or []) if c is not None]
    if sec_list:
        sec_pos = sum(1 for c in sec_list if c > 0) / len(sec_list)

    broad_up = (breadth_r is not None and breadth_r >= 0.60) or (sec_pos is not None and sec_pos >= 0.66)
    broad_dn = (breadth_r is not None and breadth_r <= 0.40) or (sec_pos is not None and sec_pos <= 0.34)
    divergence = breadth_r is not None and (
        (idx >= 0.15 and breadth_r <= 0.40) or (idx <= -0.15 and breadth_r >= 0.60)
    )

    if idx >= 0.5 and broad_up:
        regime = "TRENDING BULLISH"
        interp = "Broad participation and positive index momentum favor bullish setups."
    elif idx <= -0.5 and broad_dn:
        regime = "TRENDING BEARISH"
        interp = "Weak breadth and negative index momentum favor caution and defense."
    elif abs(idx) >= 1.0 and divergence:
        regime = "HIGH VOLATILITY"
        interp = "Large index move at odds with breadth — expect choppy, headline-driven trade."
    elif idx >= 0.15 and not broad_dn:
        regime = "RISK-ON"
        interp = "Positive index momentum; confirmation from breadth or sectors would strengthen trend."
    elif idx <= -0.15 and not broad_up:
        regime = "RISK-OFF"
        interp = "Negative index momentum; favor selectivity and tighter risk."
    else:
        regime = "MIXED / CHOPPY"
        interp = "Indexes and breadth are not aligned — setups are lower-conviction."
    return {"regime": regime, "interpretation": interp}


def compare_opportunities(
    current: List[Dict[str, Any]],
    previous: Optional[List[Dict[str, Any]]],
    *,
    min_delta: int = 3,
) -> List[Dict[str, Any]]:
    """Annotate current opportunities with movement vs the previous snapshot.

    Adds previous_score, score_delta, movement_state (NEW/RISING/FALLING/
    UNCHANGED) and status_transition (prev_status, new_status) — the latter only
    when the tier actually changed. Tiny deltas (< min_delta) count as UNCHANGED
    so noise isn't reported as movement. Safe when previous is None/empty.
    """
    prev_by: Dict[str, Dict[str, Any]] = {}
    for p in (previous or []):
        t = str(p.get("ticker") or "").upper()
        if t and t not in prev_by:  # first wins — dedupe duplicate ticker rows
            prev_by[t] = p

    out = []
    for o in (current or []):
        p = prev_by.get(str(o.get("ticker") or "").upper())
        prev_score = _to_float(p.get("score")) if p else None
        if prev_score is None:
            movement, delta = "NEW", None
        else:
            delta = int(round(o["score"] - prev_score))
            if delta >= min_delta:
                movement = "RISING"
            elif delta <= -min_delta:
                movement = "FALLING"
            else:
                movement = "UNCHANGED"
        prev_status = p.get("status") if p else None
        transition = (prev_status, o["status"]) if (prev_status and prev_status != o["status"]) else None
        out.append({
            **o,
            "previous_score": int(prev_score) if prev_score is not None else None,
            "score_delta": delta,
            "movement_state": movement,
            "previous_status": prev_status,
            "status_transition": transition,
        })
    return out


def movement_badge(opp: Dict[str, Any]) -> str:
    """Compact movement label: 'NEW' · '▲ +11' · '▼ -7' · '—'."""
    state = opp.get("movement_state")
    if state == "NEW":
        return "NEW"
    d = opp.get("score_delta")
    if state == "RISING" and d is not None:
        return f"▲ +{d}"
    if state == "FALLING" and d is not None:
        return f"▼ {d}"
    return "—"


def _rank(status: Optional[str]) -> int:
    return _STATUS_RANK.get(str(status or "").upper(), 0)


def summarize_changes(
    compared: List[Dict[str, Any]],
    previous: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """'Since last scan' rollup. Returns None when there is no previous snapshot
    to compare against (so the section is simply omitted)."""
    if not previous:
        return None
    new = [c for c in compared if c["movement_state"] == "NEW"]
    strengthened = [c for c in compared if c["movement_state"] == "RISING"]
    weakened = [c for c in compared if c["movement_state"] == "FALLING"]
    upgrades = [c for c in compared if c["status_transition"] and _rank(c["status_transition"][1]) > _rank(c["status_transition"][0])]
    downgrades = [c for c in compared if c["status_transition"] and _rank(c["status_transition"][1]) < _rank(c["status_transition"][0])]
    cur = {str(c["ticker"]).upper() for c in compared}
    dropped = [str(p.get("ticker")).upper() for p in previous
               if str(p.get("ticker") or "").upper() and str(p.get("ticker")).upper() not in cur]
    movers = [c for c in compared if c.get("score_delta") is not None]
    biggest = max(movers, key=lambda c: abs(c["score_delta"]), default=None)
    return {
        "new": new, "strengthened": strengthened, "weakened": weakened,
        "upgrades": upgrades, "downgrades": downgrades, "dropped": dropped,
        "biggest_mover": biggest,
        "any": bool(new or strengthened or weakened or upgrades or downgrades or dropped),
    }


def select_watch_next(compared: List[Dict[str, Any]], *, limit: int = 3) -> List[Dict[str, Any]]:
    """Deterministic 'what to watch next' — condition descriptions only, never
    invented price/level/catalyst values. Priority: status upgrades, then
    fading conflicts, then near-threshold WATCH names."""
    items: List[Dict[str, Any]] = []
    seen: set = set()

    def add(o, headline, detail):
        t = o["ticker"]
        if t in seen:
            return
        seen.add(t)
        items.append({"ticker": t, "headline": headline, "detail": detail,
                      "score": o.get("score"), "status": o.get("status")})

    # 1) Status upgrades — most important developing situations.
    for o in compared:
        tr = o.get("status_transition")
        if tr and _rank(tr[1]) > _rank(tr[0]):
            add(o, f"{tr[0]} → {tr[1]}", "Recently upgraded — momentum and signal confluence strengthening.")
    # 2) Fading conflicts — a fading signal against otherwise-positive confluence.
    for o in compared:
        if o.get("fading") and int(o.get("n_signals") or 0) >= 2:
            add(o, "CAUTION", "Fading signal conflicts with bullish confirmation.")
    # 3) Near-threshold WATCH names — one confirmation from STRONG.
    for o in sorted(compared, key=lambda x: x.get("score") or 0, reverse=True):
        if o.get("status") == "WATCH" and int(o.get("score") or 0) >= 65:
            add(o, f"HSF {o.get('score')}", "Would strengthen with one more confirming signal.")
    return items[: max(0, int(limit))]


def to_snapshot_rows(opps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Minimal persisted shape for movement comparison (ticker/score/status)."""
    return [{"ticker": o["ticker"], "score": o["score"], "status": o["status"]}
            for o in (opps or []) if o.get("ticker")]
