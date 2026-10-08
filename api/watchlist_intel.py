"""Watchlist Intelligence: current HSF intelligence for every symbol on a watchlist.

Read-only over the canonical data the Scanner already produced. Nothing here
scores, ranks or calls a market-data provider:

    saved market scans (db.runs)            -- written by the scheduled scans
      -> api.scans.run_opportunities(run)   -- the canonical ranked HSF setups (cached)
      -> api.today.run_df(run)              -- the raw scan rows (price, RVOL, EMA cross) (cached)
      -> market_observation()               -- one per-ticker lookup for the latest run + the
                                               previous run, cached per run pair
      -> intelligence(...) / changes(...)   -- dictionary lookups per watchlist symbol

So a watchlist of any size costs zero provider calls and no per-symbol queries;
the alert evaluator (api.alert_rules) reads the same observation object.

Fields the scans don't carry (company name, RSI, an absolute price change) are
returned as null and listed in UNAVAILABLE_FIELDS rather than estimated.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from api.today import _cached, _iso, _num, market_phase, market_runs, run_df, scan_freshness

log = logging.getLogger("hsf_api.watchlist_intel")

# A saved run never changes, so the observation for a (latest, previous) run pair
# is reused until a newer scan lands (market_runs() itself refreshes every 60 s).
OBSERVATION_TTL_S = 1800
UNAVAILABLE_FIELDS = ("company_name", "price_change", "rsi")


def _raw_fields(df: Any) -> Dict[str, Dict[str, Any]]:
    """Per ticker, the scan's own price, change, RVOL and EMA 9/21 cross. A ticker
    can appear in several scanner rows; the first non-null value wins."""
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or getattr(df, "empty", True):
        return out
    col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
    if col is None:
        return out
    pick = {"last": ("Last", "Price"), "chg_pct": ("PctChange",), "rvol": ("VolRel20",), "ema_cross": ("EMACross",)}
    cols = {k: [c for c in names if c in df.columns] for k, names in pick.items()}
    for rec in df.to_dict(orient="records"):
        t = str(rec.get(col) or "").strip().upper()
        if not t:
            continue
        cur = out.setdefault(t, {"last": None, "chg_pct": None, "rvol": None, "ema_cross": None})
        for k, names in cols.items():
            if cur[k] is not None:
                continue
            for c in names:
                v = rec.get(c)
                if k == "ema_cross":
                    if isinstance(v, str) and v.strip().lower() in ("golden", "death"):
                        cur[k] = v.strip().lower()
                        break
                else:
                    n = _num(v)
                    if n is not None:
                        cur[k] = n
                        break
    return out


def _ranked(run_id: int) -> Dict[str, Dict[str, Any]]:
    from api.scans import run_opportunities

    out: Dict[str, Dict[str, Any]] = {}
    for i, o in enumerate(run_opportunities(int(run_id))):
        t = str(o.get("ticker") or "").strip().upper()
        if t and t not in out:
            out[t] = {**o, "rank": i + 1}
    return out


def _build(cur: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from api.scans import run_opportunities

    ranked = _ranked(int(cur["id"]))
    prev_ranked = _ranked(int(prev["id"])) if prev else {}
    return {
        "observation_id": f"run:{int(cur['id'])}",
        "run_id": int(cur["id"]),
        "scan_at": cur["created_at"],
        "previous_run_id": int(prev["id"]) if prev else None,
        "previous_scan_at": prev["created_at"] if prev else None,
        "ranked": ranked,
        "total": len(ranked),
        "previous_ranked": prev_ranked,
        # The previous run's rows in rank order (the canonical events need the full list).
        "previous_rows": list(run_opportunities(int(prev["id"]))) if prev else None,
        "raw": _raw_fields(run_df(int(cur["id"]))),
    }


def market_observation() -> Optional[Dict[str, Any]]:
    """The latest full-market scan as a per-ticker lookup, plus the previous scan's
    scores and ranks. None when there is no scan yet. DatabaseUnavailable propagates."""
    runs = market_runs()
    if not runs:
        return None
    cur, prev = runs[0], (runs[1] if len(runs) > 1 else None)
    key = ("wl_obs", int(cur["id"]), int(prev["id"]) if prev else None)
    return _cached(key, lambda: _build(cur, prev), ttl_s=OBSERVATION_TTL_S)


def observation_stale(obs: Dict[str, Any], now: Optional[dt.datetime] = None) -> bool:
    """True when a scheduled full-market scan was missed (the /readyz rule)."""
    return bool(scan_freshness(obs["scan_at"], now or dt.datetime.now(dt.timezone.utc))["stale"])


def ticker_view(obs: Dict[str, Any], ticker: str, *, premium: bool) -> Dict[str, Any]:
    """One ticker's canonical state, redacted below Premium like every other surface.

    `present` = the ticker is in the latest scan at all; `ranked` = it qualified as a
    ranked HSF setup. A ticker absent from the scan has unknown state (not "inactive")."""
    from ui.entitlement_view import redact_prebreakout_opportunity

    t = ticker.upper()
    opp = obs["ranked"].get(t)
    if opp is not None and not premium:
        opp = {**redact_prebreakout_opportunity(opp, allowed=False), "rank": opp["rank"]}
    raw = obs["raw"].get(t)
    prev = obs["previous_ranked"].get(t)
    signals = [str(s) for s in (opp or {}).get("signals") or []]
    return {
        "ticker": t,
        "present": opp is not None or raw is not None,
        "ranked": opp is not None,
        "hsf_score": _num(opp.get("score")) if opp else None,
        "rank": int(opp["rank"]) if opp else None,
        "status": opp.get("status") if opp else None,
        "setup": opp.get("primary_setup") if opp else None,
        "signals": signals,
        "fading": bool(opp.get("fading")) if opp else None,
        "prebreakout": ("prebreakout" in signals) if premium else None,
        "prebreakout_score": _num(opp.get("prob")) if opp and premium else None,
        "prebreakout_rank_pct": _num(opp.get("prob_rank")) if opp and premium else None,
        "breakout_score": _num(opp.get("breakout_score")) if opp else None,
        "price": (_num(opp.get("last")) if opp and opp.get("last") is not None else (raw or {}).get("last")),
        "price_change_pct": (_num(opp.get("chg_pct")) if opp and opp.get("chg_pct") is not None
                             else (raw or {}).get("chg_pct")),
        "rvol": (_num(opp.get("rvol")) if opp and opp.get("rvol") is not None else (raw or {}).get("rvol")),
        "ema_cross": (raw or {}).get("ema_cross"),
        "previous_hsf_score": _num(prev.get("score")) if prev else None,
        "previous_rank": int(prev["rank"]) if prev else None,
    }


def _active_alert_counts(user: str, watchlist_id: int, tickers: Sequence[str]) -> Optional[Dict[str, int]]:
    """Enabled alert rules (on the ticker or on this watchlist) plus enabled
    ticker alerts, per ticker. Two queries for the whole list; None if unreadable."""
    from api import user_data
    from db import alert_rules as store

    try:
        rules = [r for r in store.list_rules(user) if r["enabled"]]
        legacy = [a for a in user_data.list_alerts(user) if a.get("enabled") and a.get("ticker")]
    except Exception as e:  # counts are a convenience; the intelligence still renders
        log.warning(json.dumps({"event": "watchlist_alert_counts_failed", "error": type(e).__name__}))
        return None
    on_list = sum(1 for r in rules if r.get("watchlist_id") == int(watchlist_id))
    out = {}
    for t in tickers:
        out[t] = (on_list + sum(1 for r in rules if r.get("ticker") == t)
                  + sum(1 for a in legacy if str(a.get("ticker")).upper() == t))
    return out


def intelligence(user: str, watchlist: Dict[str, Any], entitlements: Dict[str, bool],
                 now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """GET /v1/watchlists/{id}/intelligence."""
    t0 = time.perf_counter()
    now = now or dt.datetime.now(dt.timezone.utc)
    premium = bool(entitlements.get("can_early_breakout"))
    tickers = [str(i["ticker"]).upper() for i in watchlist.get("items") or []]
    obs: Optional[Dict[str, Any]] = None
    scan_error = None
    try:
        obs = market_observation()
    except Exception as e:  # scan data unreadable: the list still answers, marked unavailable
        scan_error = type(e).__name__
    stale = observation_stale(obs, now) if obs else None
    counts = _active_alert_counts(user, int(watchlist["id"]), tickers)
    items = []
    for item in watchlist.get("items") or []:
        t = str(item["ticker"]).upper()
        v = ticker_view(obs, t, premium=premium) if obs else None
        if v is None:
            freshness = "unavailable"
        elif not v["present"]:
            freshness = "missing"
        else:
            freshness = "stale" if stale else "fresh"
        score_change = (v["hsf_score"] - v["previous_hsf_score"]
                        if v and v["hsf_score"] is not None and v["previous_hsf_score"] is not None else None)
        rank_change = (v["previous_rank"] - v["rank"]
                       if v and v["rank"] is not None and v["previous_rank"] is not None else None)
        items.append({
            "ticker": t, "added_at": item.get("added_at"), "note": item.get("note"),
            "company_name": None, "price_change": None, "rsi": None,
            **({k: v[k] for k in ("price", "price_change_pct", "hsf_score", "previous_hsf_score", "rank",
                                  "previous_rank", "status", "setup", "signals", "fading", "prebreakout",
                                  "prebreakout_score", "prebreakout_rank_pct", "breakout_score", "rvol",
                                  "ema_cross", "ranked")} if v else {"signals": [], "ranked": False}),
            "score_change": score_change, "rank_change": rank_change,
            "in_latest_scan": bool(v and v["present"]),
            "freshness": freshness,
            "active_alert_count": counts.get(t) if counts is not None else None,
        })
    enriched = sum(1 for i in items if i["in_latest_scan"])
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(json.dumps({"event": "watchlist_intelligence", "symbols": len(items), "enriched": enriched,
                         "missing": len(items) - enriched, "ms": ms, "scan_error": scan_error}))
    return {
        "watchlist_id": int(watchlist["id"]), "name": watchlist["name"],
        "market_session": market_phase(now),
        "scan_available": obs is not None,
        "last_scan_at": _iso(obs["scan_at"]) if obs else None,
        "previous_scan_at": _iso(obs["previous_scan_at"]) if obs else None,
        "market_data_as_of": _iso(obs["scan_at"]) if obs else None,
        "stale": stale,
        "scan_total": obs["total"] if obs else None,
        "prebreakout_locked": not premium,
        "coverage": {"symbols": len(items), "enriched": enriched, "missing": len(items) - enriched},
        "unavailable_fields": list(UNAVAILABLE_FIELDS),
        "items": items,
    }


def changes(user: str, watchlist: Dict[str, Any], entitlements: Dict[str, bool]) -> Dict[str, Any]:
    """GET /v1/watchlists/{id}/changes: what changed between the previous and the latest
    market scan for this list's symbols (the canonical opportunity events: new, dropped,
    rising/falling by the canonical ±3, status changes, fading, signals added/removed incl.
    PreBreakout), each with its rank move, plus the user's rule alerts since the previous
    scan. Both scans are persisted runs, so nothing is reconstructed."""
    from analytics.opportunity_events import collapse_events, derive_opportunity_events
    from ui.entitlement_view import redact_prebreakout_rows

    premium = bool(entitlements.get("can_early_breakout"))
    tickers = [str(i["ticker"]).upper() for i in watchlist.get("items") or []]
    obs = market_observation()
    base = {"watchlist_id": int(watchlist["id"]), "name": watchlist["name"],
            "last_scan_at": _iso(obs["scan_at"]) if obs else None,
            "previous_scan_at": _iso(obs["previous_scan_at"]) if obs else None,
            "has_baseline": bool(obs and obs.get("previous_rows")), "changes": [], "alerts": []}
    if not obs or not tickers:
        return base
    watch = set(tickers)
    current = [{k: v for k, v in o.items() if k != "rank"} for t, o in obs["ranked"].items() if t in watch]
    previous = obs.get("previous_rows")
    if not premium:
        current = redact_prebreakout_rows(current, allowed=False)
        previous = redact_prebreakout_rows(previous, allowed=False) if previous else previous
    events = [e for e in derive_opportunity_events(previous, current,
                                                   previous_snapshot_time=_iso(obs["previous_scan_at"]),
                                                   current_snapshot_time=_iso(obs["scan_at"]),
                                                   source_context="watchlist")
              if str(e.get("ticker")).upper() in watch and e.get("event_type") != "VERSION_CHANGED"]
    out = []
    for e in events:
        t = str(e["ticker"]).upper()
        cur = obs["ranked"].get(t)
        prev = obs["previous_ranked"].get(t)
        out.append({
            "ticker": t, "event_type": e["event_type"], "severity": e.get("severity"),
            "previous_score": _num(e.get("previous_score")), "current_score": _num(e.get("current_score")),
            "score_delta": _num(e.get("score_delta")),
            "previous_status": e.get("previous_status"), "current_status": e.get("current_status"),
            "setup": e.get("primary_setup"), "signal": e.get("signal"),
            "previous_rank": int(prev["rank"]) if prev else None, "rank": int(cur["rank"]) if cur else None,
            "rank_change": (int(prev["rank"]) - int(cur["rank"])) if prev and cur else None,
        })
    base["changes"] = out
    base["headline"] = [{"ticker": e["ticker"], "event_type": e["event_type"]}
                        for e in collapse_events(events)]
    since = obs["previous_scan_at"] or obs["scan_at"]
    since = since if since.tzinfo else since.replace(tzinfo=dt.timezone.utc)
    try:  # same shape as /v1/alerts/events
        from api.alert_rules import list_events

        base["alerts"] = [e for e in list_events(user, limit=100, source="rule", triggered_after=since)
                          if e.get("ticker") in watch]
    except Exception as e:
        log.warning(json.dumps({"event": "watchlist_changes_alerts_failed", "error": type(e).__name__}))
    return base
