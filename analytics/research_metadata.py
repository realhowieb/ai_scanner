"""Run 57 — point-in-time research metadata for NEW observations (pure).

PERSIST WHAT THE SCANNER KNEW. DO NOT CHANGE WHAT THE SCANNER DOES.

Builds an additive `research_metadata` block for each new research observation
(CANDIDATE / NEAR_MISS / CONTROL). Every value is something the scheduled scan
already held in memory at the scan timestamp: its own rank ordering, its ranked
row's feature columns, its run configuration, the price frames' provider tags,
and the deployed commit. Nothing is fetched, recomputed, or inferred later.

Point-in-time boundary (enforced by `assert_point_in_time`):
  * never outcomes, returns, MFE/MAE, future bars, or maturation state;
  * never a regime or tier reconstructed after the fact;
  * a value the scan did not produce is stored as None with an explicit
    `*_source` explaining why. No value is fabricated or backfilled.

The block sits OUTSIDE the canonical `market`/`indicators`, so observation ids,
`data_quality` completeness, and every existing field stay byte-identical. It is
written only for new captures; historical rows are never rewritten.
"""
from __future__ import annotations

import copy
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from analytics.hsf_observation import OBSERVATION_SCHEMA_VERSION

RESEARCH_METADATA_SCHEMA = "hsf-research-meta-1.0"

# Tier: the scheduled breakout scan emits no tier. Day Trader Strong/Developing/
# Weak tiers are computed only at UI render time, from intraday fields (VWAP, ADX,
# SuperTrend, EWO) that the scheduled scan never fetches. Persisting one would
# mean recomputing it, which Run 57 forbids.
TIER_UNAVAILABLE = "TIER_NOT_EMITTED_BY_SCHEDULED_SCAN"
# Regime: `ui.opportunities.classify_market_regime` exists but only runs in the
# Market Brief UI, from SPY/QQQ closes, sector ETFs and snapshot breadth that the
# scheduled scan does not load. Capturing it would need new inputs and new calls.
REGIME_UNAVAILABLE = "REGIME_CAPTURE_UNAVAILABLE"

# The scanner's own identifiers (tied to real behavior, not invented).
SCORING_VERSION = "breakout-1"   # analytics.observation_capture breakout scanner version
RANKING_RULE = "BreakoutScore descending; CANDIDATE = top_n, NEAR_MISS = next near_miss_n"

# Ranked-row columns the scan computes and the canonical observation drops.
ROW_FEATURES = {
    "BreakoutPos20D": "breakout_pos_20d",
    "Trend20D%": "trend_20d_pct",
    "Trend10D%": "trend_10d_pct",
    "DollarVol20": "dollar_vol_20",
    "PatternTag": "pattern_tag",
    "RSvsSPY": "rs_vs_spy",
    "EMACross": "ema_cross",
}

# Anything that smells of hindsight must never appear in point-in-time metadata.
_FORBIDDEN = re.compile(
    r"(return|outcome|mfe|mae|future|matur|evaluation|horizon|hit\b|win|pnl|profit|"
    r"forward_|hindsight|realized|_label|label_)", re.I)


def _clean(v: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None; numpy scalars -> python."""
    try:
        import math
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
    except Exception:
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def commit_sha() -> Optional[str]:
    """Deployed commit when the runtime exposes one reliably (GitHub Actions)."""
    sha = (os.getenv("GITHUB_SHA") or os.getenv("HSF_COMMIT_SHA") or "").strip()
    return sha if re.fullmatch(r"[0-9a-f]{7,40}", sha) else None


def provider_summary(price_meta: Optional[Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Run-level provider mix from the price frames' own tags (no calls)."""
    sources, feeds = Counter(), Counter()
    for m in (price_meta or {}).values():
        sources[str((m or {}).get("source") or "untagged")] += 1
        if (m or {}).get("feed"):
            feeds[str(m["feed"])] += 1
    return {"price_sources": dict(sources), "price_feeds": dict(feeds)}


def build_run_context(*, universe: str, session: Optional[str], scan_id: Optional[str],
                      scan_config: Optional[Mapping[str, Any]] = None,
                      price_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
                      scan_mode: str = "scheduled") -> Dict[str, Any]:
    """Run-level point-in-time metadata shared by every cohort in one scan."""
    ctx = {
        "schema_version": RESEARCH_METADATA_SCHEMA,
        "tier_at_observation": None,
        "tier_source": TIER_UNAVAILABLE,
        "tier_version": None,
        "market_regime_at_observation": None,
        "market_regime_source": REGIME_UNAVAILABLE,
        "market_regime_version": None,
        "scoring_version": SCORING_VERSION,
        "ranking_rule": RANKING_RULE,
        "feature_schema_version": OBSERVATION_SCHEMA_VERSION,
        "scanner_commit_sha": commit_sha(),
        "universe_name": str(universe) if universe else None,
        "universe_version": str(universe) if universe else None,
        "scan_mode": scan_mode,
        "session": session,
        "scan_id": scan_id,
        "scan_config": {k: _clean(v) for k, v in (scan_config or {}).items()},
        **provider_summary(price_meta),
    }
    assert_point_in_time(ctx)
    return ctx


def observation_metadata(run_context: Mapping[str, Any], *, symbol: str,
                         rank: Optional[int] = None,
                         row: Optional[Mapping[str, Any]] = None,
                         price_meta: Optional[Mapping[str, Mapping[str, Any]]] = None
                         ) -> Dict[str, Any]:
    """Per-observation block = run context + what the scan knew about this row."""
    meta = copy.deepcopy(dict(run_context))
    for k in ("price_sources", "price_feeds"):
        meta.pop(k, None)
    pm = (price_meta or {}).get(str(symbol).upper()) or {}
    meta["rank_at_observation"] = int(rank) if rank is not None else None
    meta["price_provider"] = pm.get("source")
    meta["price_feed"] = pm.get("feed")
    feats = {}
    for col, name in ROW_FEATURES.items():
        if row is not None and col in row:
            v = _clean(row.get(col))
            if v is not None:
                feats[name] = v
    meta["row_features"] = feats
    assert_point_in_time(meta)
    return meta


def attach(observations: Sequence[Dict[str, Any]], run_context: Optional[Mapping[str, Any]], *,
           rows: Optional[Sequence[Mapping[str, Any]]] = None, rank_offset: Optional[int] = 0,
           price_meta: Optional[Mapping[str, Mapping[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Attach metadata in place and return the list. Best-effort and NON-FATAL:
    on any error the observations are returned untouched (capture proceeds).

    `rows` aligns 1:1 with `observations` in ranked order; `rank_offset=None`
    means the cohort is unranked (CONTROL)."""
    if not run_context:
        return list(observations)
    try:
        by_symbol = {}
        for i, r in enumerate(rows or []):
            s = str(r.get("Ticker") or r.get("Symbol") or "").upper()
            if s and s not in by_symbol:
                by_symbol[s] = (i, r)
        metas = []
        for o in observations:
            sym = str(o.get("symbol") or "").upper()
            idx, row = by_symbol.get(sym, (None, None))
            rank = (rank_offset + idx + 1) if (rank_offset is not None and idx is not None) else None
            metas.append(observation_metadata(run_context, symbol=sym, rank=rank, row=row,
                                              price_meta=price_meta))
        for o, m in zip(observations, metas):
            o["research_metadata"] = m
    except Exception:
        pass
    return list(observations)


def forbidden_keys(obj: Any, path: str = "") -> List[str]:
    bad: List[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if _FORBIDDEN.search(str(k)):
                bad.append(f"{path}.{k}")
            bad += forbidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += forbidden_keys(v, f"{path}[{i}]")
    return bad


def assert_point_in_time(meta: Mapping[str, Any]) -> None:
    """Reject any key that could carry hindsight (outcomes, returns, labels)."""
    bad = forbidden_keys(meta)
    if bad:
        raise ValueError(f"point-in-time metadata may not contain hindsight fields: {bad[:5]}")


def coverage(observations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Share of observations carrying each metadata field (completeness only)."""
    fields = ("tier_at_observation", "market_regime_at_observation", "scoring_version",
              "scanner_commit_sha", "universe_name", "price_provider", "scan_mode",
              "rank_at_observation")
    obs = list(observations)
    n = len(obs)
    has_block = sum(1 for o in obs if isinstance(o.get("research_metadata"), Mapping))
    out: Dict[str, Any] = {"observations": n, "with_metadata_block": has_block}
    for f in fields:
        k = sum(1 for o in obs if (o.get("research_metadata") or {}).get(f) not in (None, ""))
        out[f] = {"present": k, "pct": round(100.0 * k / n, 2) if n else None}
    return out
