"""Canonical HSF research dataset: point-in-time observations (pure, no I/O).

One research observation = one frozen HSF opportunity row in
``signal_outcomes`` (source='opportunity'), the same row and id that Outcome
Intelligence measures. Its point-in-time features come from two persisted,
immutable places only:

  1. the opportunity row's own frozen payload (HSF score, components, setup,
     status, signals, PreBreakout %), and
  2. the scheduled scan record in ``hsf_observations`` for the same ticker,
     joined BACKWARD in time (see ``match_scan_observation``).

Nothing is fetched from a provider, recomputed from today's scanner, or filled
with a default. Labels (returns, MFE/MAE, SPY) are read into a separate
``OutcomeRecord``. ``build_dataset`` assembles a deterministic dataset with a
fingerprint so the same inputs always produce the same bytes.

Temporal join rule for scan features (the only join in this layer):
    candidate scan records = same ticker, context 'scheduled:*',
                             known_at <= observed_at  (known_at = the row's write time)
                             observed_at - scan_timestamp <= MAX_SCAN_LAG
    pick the latest scan_timestamp; ties -> 'scheduled:us_market' first, then
    the lowest observation id. Never "ticker -> latest record".
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analytics import research_schema as rs

MAX_SCAN_LAG = _dt.timedelta(hours=3)
PREFERRED_SCAN_CONTEXT = "scheduled:us_market"
DATASET_PREFIX = "hsf-ml"

JOIN_MATCHED = "MATCHED"
JOIN_STALE = "NO_SCAN_WITHIN_LAG"      # an earlier scan exists, but older than MAX_SCAN_LAG
JOIN_MISSING = "NO_SCAN_RECORD"        # no eligible scan record at all
JOIN_FUTURE_ONLY = "ONLY_LATER_SCANS"  # records exist for the ticker, all written after observed_at

# Maturity states are Outcome Intelligence's own (one definition in the codebase).
from analytics.outcome_intelligence import INVALID, MATURED, PENDING, UNAVAILABLE  # noqa: E402


# --------------------------------------------------------------------------- parsing
def to_dt(value: Any) -> Optional[_dt.datetime]:
    """Aware UTC datetime from a datetime / ISO string / SQLite 'YYYY-MM-DD HH:MM:SS'."""
    if value is None:
        return None
    try:
        d = value if isinstance(value, _dt.datetime) else _dt.datetime.fromisoformat(
            str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.astimezone(_dt.timezone.utc)


def iso(value: Any) -> Optional[str]:
    d = to_dt(value)
    return d.isoformat() if d is not None else None


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(v: Any) -> Optional[int]:
    f = _num(v)
    return int(f) if f is not None else None


def _str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _json(v: Any) -> Dict[str, Any]:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return {}
    return v if isinstance(v, dict) else {}


def _bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


# --------------------------------------------------------------------------- observation
def observation_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Identity + provenance + point-in-time HSF state of one opportunity row.
    Reads only the frozen payload; never an outcome column."""
    raw = _json(row.get("raw_signal"))
    ind = _json(row.get("indicators"))
    return {
        "observation_id": int(row["id"]),
        "ticker": str(row.get("ticker") or "").upper(),
        "observed_at": iso(row.get("fired_at")),
        "hsf_score": _num(raw.get("hsf_score")),
        "setup": _str(raw.get("primary_setup") or ind.get("primary_setup")),
        "status": _str(raw.get("status") or ind.get("status")),
        "signals": sorted(str(s) for s in (ind.get("signals") or [])),
        "scoring_version": _str(raw.get("score_version")),
    }


def snapshot_ranks(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Tuple[int, int]]:
    """{observation_id: (rank, size)} within each frozen snapshot (same fired_at).

    Uses only rows frozen at that exact instant, so it is point-in-time safe.
    Rank 1 = highest HSF Score; ties by ticker then id (the original list order
    was not stored, so this is a documented reconstruction)."""
    groups: Dict[str, List[Tuple[float, str, int]]] = defaultdict(list)
    for r in rows:
        ts = iso(r.get("fired_at"))
        if ts is None or r.get("id") is None:
            continue
        score = _num(_json(r.get("raw_signal")).get("hsf_score"))
        groups[ts].append((-(score if score is not None else -1.0), str(r.get("ticker") or "").upper(), int(r["id"])))
    out: Dict[int, Tuple[int, int]] = {}
    for members in groups.values():
        members.sort()
        for i, (_s, _t, oid) in enumerate(members):
            out[oid] = (i + 1, len(members))
    return out


def scan_record_view(scan_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one hsf_observations row (observation_id, symbol, context,
    record, created_at) into what the join and the snapshot need."""
    rec = _json(scan_row.get("record"))
    return {
        "scan_observation_id": str(scan_row.get("observation_id") or rec.get("observation_id") or ""),
        "symbol": str(scan_row.get("symbol") or rec.get("symbol") or "").upper(),
        "context": str(scan_row.get("context") or rec.get("context") or ""),
        "scan_timestamp": to_dt(rec.get("scan_timestamp") or rec.get("timestamp") or scan_row.get("timestamp")),
        "known_at": to_dt(scan_row.get("created_at")),
        "record": rec,
    }


def index_scan_records(scan_rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scan_rows:
        v = scan_record_view(r)
        if v["symbol"] and v["scan_timestamp"] is not None:
            by_symbol[v["symbol"]].append(v)
    return by_symbol


def match_scan_observation(ticker: str, observed_at: Any, by_symbol: Mapping[str, Sequence[Dict[str, Any]]],
                           *, max_lag: _dt.timedelta = MAX_SCAN_LAG) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Backward as-of join (module docstring). Returns (scan view or None, join info).

    A scan record is eligible only when it was both started AND written at or
    before observed_at. A record without a write time (known_at) is accepted on
    scan_timestamp alone and flagged, because the write time is the only proof
    the values existed before the observation."""
    obs = to_dt(observed_at)
    cands = list(by_symbol.get(str(ticker).upper(), ()))
    info: Dict[str, Any] = {"status": JOIN_MISSING, "scan_observation_id": None, "scan_timestamp": None,
                            "scan_context": None, "lag_seconds": None, "known_at_verified": None,
                            "max_lag_seconds": int(max_lag.total_seconds())}
    if obs is None:
        return None, info
    eligible = [c for c in cands if c["scan_timestamp"] <= obs
                and (c["known_at"] is None or c["known_at"] <= obs)]
    if not eligible:
        if cands:
            info["status"] = JOIN_FUTURE_ONLY
        return None, info
    top = max((c["scan_timestamp"], c["context"] == PREFERRED_SCAN_CONTEXT) for c in eligible)
    best = min((c for c in eligible if (c["scan_timestamp"], c["context"] == PREFERRED_SCAN_CONTEXT) == top),
               key=lambda c: c["scan_observation_id"])
    lag = obs - best["scan_timestamp"]
    if lag > max_lag:
        info.update(status=JOIN_STALE, lag_seconds=int(lag.total_seconds()))
        return None, info
    info.update(status=JOIN_MATCHED, scan_observation_id=best["scan_observation_id"],
                scan_timestamp=best["scan_timestamp"].isoformat(), scan_context=best["context"],
                lag_seconds=int(lag.total_seconds()), known_at_verified=best["known_at"] is not None)
    return best, info


def _scanner(rec: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for s in rec.get("scanners") or []:
        if isinstance(s, dict) and s.get("name") == name:
            return s
    return {}


def feature_snapshot(row: Mapping[str, Any], scan: Optional[Mapping[str, Any]], join_info: Mapping[str, Any],
                     *, rank: Optional[Tuple[int, int]] = None,
                     feature_schema_version: int = rs.FEATURE_SCHEMA_VERSION) -> rs.FeatureSnapshot:
    """Build the FeatureSnapshot for one opportunity row. Pure: reads the frozen
    payload and the already-joined scan record; never an outcome column."""
    rs.feature_schema(feature_schema_version)  # validates the version
    raw = _json(row.get("raw_signal"))
    ind = _json(row.get("indicators"))
    comps = raw.get("score_components") or {}
    v: Dict[str, Any] = {
        "hsf_score": _num(raw.get("hsf_score")),
        "hsf_score_version": _str(raw.get("score_version")),
        "hsf_signals_component": _num(comps.get("signals_component")),
        "hsf_model_component": _num(comps.get("model_component")),
        "hsf_momentum_component": _num(comps.get("momentum_component")),
        "hsf_fading_penalty": _num(comps.get("fading_penalty")),
        "primary_setup": _str(raw.get("primary_setup") or ind.get("primary_setup")),
        "hsf_status": _str(raw.get("status") or ind.get("status")),
        "signals": sorted(str(s) for s in (ind.get("signals") or [])),
        "n_signals": _int(ind.get("n_signals")),
        "fading": _bool(ind.get("fading")),
        "chg_pct": _num(ind.get("chg_pct")),
        "gap_pct": _num(ind.get("gap_pct")),
        "breakout_score": _num(row.get("setup_score")),
        "prebreakout_prob": _num(row.get("prebreakout_prob")),
        "snapshot_rank": rank[0] if rank else None,
        "snapshot_size": rank[1] if rank else None,
    }
    rec = (scan or {}).get("record") or {}
    mkt, sind = rec.get("market") or {}, rec.get("indicators") or {}
    meta = rec.get("research_metadata") or {}
    feats = meta.get("row_features") or {}
    bo = _scanner(rec, "breakout")
    v.update({
        "price": _num(mkt.get("price")),
        "volume": _num(mkt.get("volume")),
        "rvol_20": _num(sind.get("rvol")),
        "volatility_20d_pct": _num(sind.get("atr_pct")),
        "scan_gap_pct": _num(sind.get("gap_pct")),
        "scan_chg_pct": _num(sind.get("chg_pct")),
        "scanner_breakout_score": _num(bo.get("score")),
        "is_breakout": _bool((bo.get("meta") or {}).get("is_breakout")) if bo else None,
        "trend_10d_pct": _num(feats.get("trend_10d_pct")),
        "trend_20d_pct": _num(feats.get("trend_20d_pct")),
        "breakout_pos_20d": _num(feats.get("breakout_pos_20d")),
        "dollar_vol_20": _num(feats.get("dollar_vol_20")),
        "rs_vs_spy": _num(feats.get("rs_vs_spy")),
        "ema_cross": _str(feats.get("ema_cross")),
        "pattern_tag": _str(feats.get("pattern_tag")),
        "scanner_rank": _int(meta.get("rank_at_observation")),
    })
    names = set(rs.feature_names(feature_schema_version))
    return rs.FeatureSnapshot(
        observation_id=int(row["id"]), ticker=str(row.get("ticker") or "").upper(),
        observed_at=iso(row.get("fired_at")) or "", feature_schema_version=int(feature_schema_version),
        values={k: val for k, val in v.items() if k in names}, join=dict(join_info))


def scan_provenance(scan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Model/scan provenance recorded by the scan itself (never today's deployment)."""
    rec = (scan or {}).get("record") or {}
    meta = rec.get("research_metadata") or {}
    versions = rec.get("versions") or {}
    return {
        "scan_id": _str(meta.get("scan_id") or (rec.get("market_context") or {}).get("scan_id")),
        "universe": _str(meta.get("universe_name") or rec.get("universe_version")),
        "scanner_scoring_version": _str(meta.get("scoring_version")),
        "scanner_commit_sha": _str(meta.get("scanner_commit_sha")),
        "observation_schema_version": _str(rec.get("schema_version")),
        # Code constant at capture time. NOT the served model (that lives in the
        # loaded bundle and was never frozen), so model_version stays null.
        "prebreakout_model_code_constant": _str(versions.get("prebreakout_model")),
    }


# --------------------------------------------------------------------------- outcomes
def entry_day(observed_at: Any) -> Optional[_dt.date]:
    """Entry bar of the canonical outcome: the first trading day on/after the UTC
    date of fired_at (analytics.signal_outcomes._entry_position)."""
    from analytics import market_calendar as mc

    d = to_dt(observed_at)
    if d is None:
        return None
    day = d.date()
    for _ in range(10):
        if mc.is_trading_day(day):
            return day
        day += _dt.timedelta(days=1)
    return None


def label_window_end(start: Optional[_dt.date], days: int = 5) -> Optional[_dt.date]:
    """The trading day `days` bars after the entry bar (last bar any label reads)."""
    from analytics import market_calendar as mc

    if start is None:
        return None
    d, n = start, 0
    while n < days:
        d += _dt.timedelta(days=1)
        if mc.is_trading_day(d):
            n += 1
    return d


def outcome_record(row: Mapping[str, Any], *,
                   label_schema_version: int = rs.LABEL_SCHEMA_VERSION) -> rs.OutcomeRecord:
    """Labels for one observation, taken from the canonical Outcome Intelligence
    record (analytics.outcome_intelligence.canonical_record): same per-horizon
    maturity (pending / matured / unavailable / invalid), same SPY benchmark and
    excess return, same certification rule. Only the shape differs: a frozen
    OutcomeRecord that can't be mistaken for a feature payload."""
    from analytics import outcome_intelligence as oi

    rs.label_schema(label_schema_version)
    canon = oi.canonical_record(dict(row))
    vals: Dict[str, Any] = {}
    maturity: Dict[str, str] = {}
    for h in rs.HORIZONS_DAYS:
        ho = canon.horizons[h]
        maturity[f"{h}d"] = ho.status
        vals[f"return_{h}d"] = ho.raw_return
        vals[f"benchmark_return_{h}d"] = ho.benchmark_return
        vals[f"excess_return_{h}d"] = ho.excess_return
    vals["mfe_5d"] = canon.horizons[5].mfe
    vals["mae_5d"] = canon.horizons[5].mae
    start = entry_day(row.get("fired_at"))
    names = set(rs.label_names(label_schema_version))
    return rs.OutcomeRecord(
        observation_id=int(row["id"]), ticker=str(row.get("ticker") or "").upper(),
        observed_at=iso(row.get("fired_at")) or "", label_schema_version=int(label_schema_version),
        values={k: val for k, val in vals.items() if k in names}, maturity=maturity,
        certified=bool(canon.certified),
        outcome_computed_at=iso(row.get("outcome_computed_at")),
        entry_day=start.isoformat() if start else None,
        label_window_end=(label_window_end(start).isoformat() if start else None))


# --------------------------------------------------------------------------- records + filters
def build_records(rows: Sequence[Mapping[str, Any]], scan_rows: Iterable[Mapping[str, Any]], *,
                  feature_schema_version: int = rs.FEATURE_SCHEMA_VERSION,
                  label_schema_version: int = rs.LABEL_SCHEMA_VERSION,
                  max_lag: _dt.timedelta = MAX_SCAN_LAG) -> List[Dict[str, Any]]:
    """One research record per opportunity row, sorted (observed_at, id).

    Each record keeps three separate objects: `observation` (identity,
    provenance, point-in-time HSF state), `features` (FeatureSnapshot) and
    `outcome` (OutcomeRecord). Ranks are computed over every row given (pass
    whole snapshots), so filtering afterwards never changes a rank."""
    by_symbol = index_scan_records(scan_rows)
    ranks = snapshot_ranks(rows)
    groups = overlap_groups(rows)
    out = []
    for row in rows:
        if row.get("id") is None:
            continue
        ident = observation_identity(row)
        scan, join_info = match_scan_observation(ident["ticker"], row.get("fired_at"), by_symbol, max_lag=max_lag)
        snap = feature_snapshot(row, scan, join_info, rank=ranks.get(ident["observation_id"]),
                                feature_schema_version=feature_schema_version)
        outc = outcome_record(row, label_schema_version=label_schema_version)
        g = groups.get(ident["observation_id"], {})
        observation = {
            **ident,
            "source": "opportunity",
            "provenance": {**scan_provenance(scan), "run_id": None, "model_version": None,
                           "feature_schema_version": snap.feature_schema_version,
                           "label_schema_version": outc.label_schema_version},
            "rank": snap.values.get("snapshot_rank"),
            "maturity": dict(outc.maturity),
            "certified": outc.certified,
            "overlap": g,
        }
        out.append({"observation": observation, "features": snap, "outcome": outc})
    out.sort(key=lambda r: (r["observation"]["observed_at"] or "", r["observation"]["observation_id"]))
    return out


FILTER_KEYS = ("start_date", "end_date", "ticker", "setup", "min_score", "max_score", "horizon",
               "matured_only", "certified_only", "scoring_version", "model_version")


def normalize_filters(**kw: Any) -> Dict[str, Any]:
    """Every filter explicit, with None for 'not filtered'. Validates values."""
    f = {k: kw.get(k) for k in FILTER_KEYS}
    for k in ("start_date", "end_date"):
        if f[k] is not None and not isinstance(f[k], _dt.date):
            f[k] = _dt.date.fromisoformat(str(f[k]))
        if isinstance(f[k], _dt.datetime):
            f[k] = f[k].date()
    if f["start_date"] and f["end_date"] and f["start_date"] > f["end_date"]:
        raise ValueError("start_date must be on or before end_date")
    if f["horizon"] is not None:
        f["horizon"] = int(f["horizon"])
        if f["horizon"] not in rs.HORIZONS_DAYS:
            raise ValueError(f"horizon must be one of {list(rs.HORIZONS_DAYS)} trading days")
    for k in ("min_score", "max_score"):
        if f[k] is not None:
            f[k] = float(f[k])
    if f["ticker"] is not None:
        f["ticker"] = str(f["ticker"]).strip().upper() or None
    f["matured_only"] = bool(f["matured_only"])
    f["certified_only"] = bool(f["certified_only"])
    return f


def _matches(rec: Mapping[str, Any], f: Mapping[str, Any]) -> bool:
    o = rec["observation"]
    d = to_dt(o["observed_at"])
    if f.get("start_date") and (d is None or d.date() < f["start_date"]):
        return False
    if f.get("end_date") and (d is None or d.date() > f["end_date"]):
        return False
    if f.get("ticker") and o["ticker"] != f["ticker"]:
        return False
    if f.get("setup") and (o.get("setup") or "").lower() != str(f["setup"]).lower():
        return False
    score = o.get("hsf_score")
    if f.get("min_score") is not None and (score is None or score < f["min_score"]):
        return False
    if f.get("max_score") is not None and (score is None or score > f["max_score"]):
        return False
    if f.get("scoring_version") is not None and (o.get("scoring_version") or "UNKNOWN") != str(f["scoring_version"]):
        return False
    if f.get("model_version") is not None and (o["provenance"].get("model_version") or "UNKNOWN") != str(f["model_version"]):
        return False
    if f.get("matured_only"):
        h = f.get("horizon") or 5
        if o["maturity"].get(f"{h}d") != MATURED:
            return False
    if f.get("certified_only") and not o.get("certified"):
        return False
    return True


def apply_filters(records: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [r for r in records if _matches(r, filters)]


# --------------------------------------------------------------------------- overlap / quality / coverage
def overlap_groups(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Purge/embargo metadata. Rows of one ticker that share an entry day share
    the identical outcome window, so they are one overlap group. Nothing is
    removed; the ML v3 audit decides how to purge."""
    by_key: Dict[Tuple[str, Optional[str]], List[int]] = defaultdict(list)
    meta: Dict[int, Tuple[str, Optional[_dt.date]]] = {}
    for r in rows:
        if r.get("id") is None:
            continue
        t = str(r.get("ticker") or "").upper()
        e = entry_day(r.get("fired_at"))
        meta[int(r["id"])] = (t, e)
        by_key[(t, e.isoformat() if e else None)].append(int(r["id"]))
    windows: Dict[str, List[Tuple[_dt.date, _dt.date]]] = defaultdict(list)
    for (t, e), ids in by_key.items():
        if e:
            s = _dt.date.fromisoformat(e)
            windows[t].append((s, label_window_end(s)))
    out = {}
    for oid, (t, e) in meta.items():
        key = (t, e.isoformat() if e else None)
        ids = sorted(by_key[key])
        overlapping_days = 0
        if e:
            end = label_window_end(e)
            overlapping_days = sum(1 for s, en in windows[t] if s != e and s <= end and en >= e)
        out[oid] = {"group": f"{t}|{key[1]}", "group_size": len(ids), "first_in_group": ids[0] == oid,
                    "overlapping_entry_days": overlapping_days}
    return out


def _score_ok(v: Optional[float]) -> bool:
    return v is not None and 0.0 <= v <= 100.0


def quality_report(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Classify suspicious data; never delete it."""
    n = len(records)
    dup_keys = Counter((str(r.get("ticker") or "").upper(), iso(r.get("fired_at"))) for r in rows)
    duplicates = sum(c - 1 for c in dup_keys.values() if c > 1)
    missing_ts = sum(1 for r in rows if to_dt(r.get("fired_at")) is None)
    invalid_scores = sum(1 for r in records if not _score_ok(r["observation"]["hsf_score"]))
    prices = [r["features"].values.get("price") for r in records]
    invalid_prices = sum(1 for p in prices if p is not None and p <= 0)
    matured = [r for r in records if r["outcome"].maturity.get("5d") == MATURED]
    bench_gaps = sum(1 for r in matured if r["outcome"].values.get("benchmark_return_5d") is None)
    groups = Counter(r["observation"]["overlap"].get("group") for r in records)
    same_ticker_day_obs = sum(c for c in groups.values() if c > 1)
    overlapping = sum(1 for r in records if (r["observation"]["overlap"].get("overlapping_entry_days") or 0) > 0)
    unknown_version = sum(1 for r in records if r["observation"]["scoring_version"] is None)
    scores = [r["observation"]["hsf_score"] for r in records if r["observation"]["hsf_score"] is not None]
    buckets = Counter(_bucket(s) for s in scores)
    joins = Counter(r["features"].join.get("status") for r in records)
    dates = Counter((r["observation"]["observed_at"] or "")[:10] for r in records)
    return {
        "observations": n,
        "duplicate_observations": duplicates,
        "missing_timestamps": missing_ts,
        "invalid_scores": invalid_scores,
        "invalid_prices": invalid_prices,
        "missing_outcomes": {"pending": sum(1 for r in records if r["outcome"].maturity.get("5d") == PENDING),
                             "unavailable": sum(1 for r in records if r["outcome"].maturity.get("5d") == UNAVAILABLE),
                             "invalid": sum(1 for r in records if r["outcome"].maturity.get("5d") == INVALID)},
        "benchmark_gaps_among_matured_5d": bench_gaps,
        "overlap": {"rows_in_multi_row_ticker_day_groups": same_ticker_day_obs,
                    "ticker_day_groups": len(groups),
                    "rows_with_overlapping_label_windows": overlapping},
        "unknown_scoring_version": unknown_version,
        "scan_join_status": dict(sorted((str(k), v) for k, v in joins.items())),
        "ticker_count": len({r["observation"]["ticker"] for r in records}),
        "date_count": len(dates),
        "setups": dict(sorted(Counter(r["observation"]["setup"] or "UNKNOWN" for r in records).items())),
        "scoring_versions": dict(sorted(Counter(r["observation"]["scoring_version"] or "UNKNOWN"
                                                for r in records).items())),
        "model_versions": dict(sorted(Counter(r["observation"]["provenance"]["model_version"] or "UNKNOWN"
                                              for r in records).items())),
        "score_distribution": dict(sorted(buckets.items())),
        "maturity_5d": dict(sorted(Counter(r["outcome"].maturity.get("5d") for r in records).items())),
    }


def _bucket(score: float) -> str:
    from analytics.hsf_calibration import _bucket_of

    return _bucket_of(score) or "out_of_range"


def _rate(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def coverage(records: Sequence[Mapping[str, Any]], *,
             feature_schema_version: int = rs.FEATURE_SCHEMA_VERSION) -> Dict[str, Any]:
    """Real coverage from the records given. Rates are share of observations
    with a non-null value; outcome rates for labels are share of observations
    whose cron has scored them (pending rows can't have labels yet)."""
    n = len(records)
    computed = [r for r in records if r["outcome"].maturity.get("5d") not in (PENDING, None)]
    feats = {}
    for name in rs.feature_names(feature_schema_version):
        k = 0
        for r in records:
            v = r["features"].values.get(name)
            if v is not None and v != ():
                k += 1
        feats[name] = _rate(k, n)
    labels = {}
    for name in rs.label_names():
        k = sum(1 for r in computed if r["outcome"].values.get(name) is not None)
        labels[name] = {"present": k, "of_scored": len(computed), "rate": _rate(k, len(computed))}
    times = sorted(r["observation"]["observed_at"] for r in records if r["observation"]["observed_at"])
    mat = Counter(r["outcome"].maturity.get("5d") for r in records)
    joined = sum(1 for r in records if r["features"].join.get("status") == JOIN_MATCHED)
    meta_present = sum(1 for r in records if r["features"].values.get("trend_20d_pct") is not None
                       or r["features"].values.get("scanner_rank") is not None)
    return {
        "total_observations": n,
        "matured_observations": mat.get(MATURED, 0),
        "pending_observations": mat.get(PENDING, 0),
        "unavailable_observations": mat.get(UNAVAILABLE, 0),
        "invalid_observations": mat.get(INVALID, 0),
        "certified_observations": sum(1 for r in records if r["outcome"].certified),
        "earliest_observation": times[0] if times else None,
        "latest_observation": times[-1] if times else None,
        "horizons": {f"{h}d": labels[f"return_{h}d"] for h in rs.HORIZONS_DAYS},
        "benchmark": {f"{h}d": labels[f"benchmark_return_{h}d"] for h in rs.HORIZONS_DAYS},
        "excess": {f"{h}d": labels[f"excess_return_{h}d"] for h in rs.HORIZONS_DAYS},
        "mfe_5d": labels["mfe_5d"],
        "mae_5d": labels["mae_5d"],
        "unsupported_horizons": {"10_bar": None, "15_bar": None, "20_bar": None,
                                 "note": "No 10/15/20-bar outcomes exist for HSF observations; not fabricated."},
        "features": feats,
        "unavailable_features": [{"name": u["name"], "coverage": 0.0, "classification": u["pit"]}
                                 for u in rs.UNAVAILABLE_FEATURES],
        "scan_feature_join_rate": _rate(joined, n),
        "research_metadata_rate": _rate(meta_present, n),
        "market_context": {"rs_vs_spy": feats.get("rs_vs_spy"), "spy_trend": 0.0, "qqq_trend": 0.0,
                           "market_regime": 0.0, "breadth": 0.0, "sector": 0.0},
        "scoring_versions": dict(sorted(Counter(r["observation"]["scoring_version"] or "UNKNOWN"
                                                for r in records).items())),
        "model_versions": dict(sorted(Counter(r["observation"]["provenance"]["model_version"] or "UNKNOWN"
                                              for r in records).items())),
        "feature_schema_version": int(feature_schema_version),
        "label_schema_version": rs.LABEL_SCHEMA_VERSION,
    }


# --------------------------------------------------------------------------- dataset
def _canon(v: Any) -> Any:
    if isinstance(v, float):
        return float(repr(v)) if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return v


def fingerprint(payload: Mapping[str, Any]) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace, UTF-8)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_dataset(rows: Sequence[Mapping[str, Any]], scan_rows: Iterable[Mapping[str, Any]], *,
                  filters: Optional[Mapping[str, Any]] = None,
                  feature_schema_version: int = rs.FEATURE_SCHEMA_VERSION,
                  label_schema_version: int = rs.LABEL_SCHEMA_VERSION,
                  members: Optional[Iterable[int]] = None,
                  code_revision: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic research dataset (no training, no I/O).

    Output: metadata, ordered observation ids + timestamps, the feature matrix
    (columns = feature schema order), the label matrix (columns = label schema
    order), per-row maturity, and a fingerprint over all of it. `members`
    restricts to a finalized version's ids. Created-at and the version name are
    deliberately NOT part of the fingerprint, so a rebuild of unchanged data
    from unchanged code yields the same fingerprint."""
    f = normalize_filters(**(filters or {}))
    records = build_records(rows, scan_rows, feature_schema_version=feature_schema_version,
                            label_schema_version=label_schema_version)
    records = apply_filters(records, f)
    if members is not None:
        keep = {int(m) for m in members}
        records = [r for r in records if r["observation"]["observation_id"] in keep]
    fnames = list(rs.feature_names(feature_schema_version))
    lnames = list(rs.label_names(label_schema_version))
    ids = [r["observation"]["observation_id"] for r in records]
    obs_at = [r["observation"]["observed_at"] for r in records]
    X = [[_canon(v) for v in r["features"].vector()] for r in records]
    Y = [[_canon(v) for v in r["outcome"].vector()] for r in records]
    maturity = [r["outcome"].maturity.get("5d") for r in records]
    filt = {k: _canon(v) for k, v in f.items()}
    fp = fingerprint({"feature_schema_version": int(feature_schema_version),
                      "label_schema_version": int(label_schema_version), "filters": filt,
                      "feature_columns": fnames, "label_columns": lnames, "observation_ids": ids,
                      "observed_at": obs_at, "features": X, "labels": Y, "maturity": maturity})
    cov = coverage(records, feature_schema_version=feature_schema_version)
    quality = quality_report(rows, records)
    metadata = {
        "feature_schema_version": int(feature_schema_version),
        "label_schema_version": int(label_schema_version),
        "filters": filt,
        "observation_count": len(ids),
        "matured_count": cov["matured_observations"],
        "certified_count": cov["certified_observations"],
        "observation_range": {"earliest": cov["earliest_observation"], "latest": cov["latest_observation"]},
        "scoring_versions": cov["scoring_versions"],
        "model_versions": cov["model_versions"],
        "data_quality": quality,
        "feature_coverage": cov["features"],
        "code_revision": code_revision,
        "fingerprint": fp,
        "fingerprint_method": "sha256 over canonical JSON of schema versions, filters, column names, "
                              "ordered observation ids + observed_at, feature and label matrices, 5d maturity",
    }
    return {"metadata": metadata, "observation_ids": ids, "observed_at": obs_at,
            "feature_columns": fnames, "features": X, "label_columns": lnames, "labels": Y,
            "maturity_5d": maturity, "records": records}


def version_name(day: _dt.date, existing: Iterable[str]) -> str:
    """hsf-ml-YYYY-MM-DD-vN with the next free N for that day."""
    prefix = f"{DATASET_PREFIX}-{day.isoformat()}-v"
    used = set()
    for name in existing:
        if str(name).startswith(prefix):
            try:
                used.add(int(str(name)[len(prefix):]))
            except ValueError:
                continue
    n = 1
    while n in used:
        n += 1
    return f"{prefix}{n}"
