"""Custom scans for API clients: plan rules and the scan itself.

`plan_scan()` turns a request into the exact parameters the web's Custom Scan
page would use, enforcing every plan rule on the server (the web enforces them
by disabling widgets, which an API client can't be trusted to do):

  universe   sp500 (every plan) · nasdaq, combo (Pro+) · us_market (Premium+)
             · watchlist (your own) · ticker (every plan)
  rows       top_n up to the plan's Scanner rows (Free 25, Pro 100, Premium 200)
  sessions   premarket / afterhours: Pro+ (outside that session the web scans
             the regular session instead, and so does this)
  filters    unusual volume and the gap filter: Pro+
  caps       NASDAQ / Combo ticker caps: Pro up to 4,000 / 6,000; Premium and
             admin scan the full lists (the caps are ignored)

`run_scan()` performs it with the web's code: resolve_scan_universe (same
loaders and liquidity pre-filter) and run_manual_scan_execution with
run_breakout_scan, then shapes rows exactly like GET /v1/scans/latest.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

UNIVERSES = ("sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker")
MARKET = {"sp500": "SP500", "nasdaq": "NASDAQ", "combo": "COMBO", "us_market": "US_MARKET"}
LABELS = {"sp500": "SP500", "nasdaq": "NASDAQ", "combo": "Combo", "us_market": "US Market"}
UNIVERSE_FEATURE = {"sp500": "can_scan_sp500", "nasdaq": "can_scan_nasdaq",
                    "combo": "can_scan_nasdaq", "us_market": "can_full_universe"}
PRO_NASDAQ_CAP, PRO_COMBO_CAP = 4000, 6000          # ui.filters: Pro caps
DEFAULT_NASDAQ, DEFAULT_COMBO = 1200, 1000          # ui.filters defaults


class PlanError(PermissionError):
    """The plan doesn't include what was asked for (403)."""


def _need(ent: Dict[str, bool], feature: str, message: str) -> None:
    if not ent.get(feature):
        raise PlanError(message)


def plan_scan(req: Dict[str, Any], *, entitlements: Dict[str, bool], tier: str, is_admin: bool,
              max_results: int, now_session: str) -> Dict[str, Any]:
    """Validated, effective scan parameters (raises PlanError or ValueError)."""
    ent = dict(entitlements)
    universe = req["universe"]
    f = dict(req.get("filters") or {})
    full_lists = bool(is_admin or ent.get("can_full_universe"))

    if universe in UNIVERSE_FEATURE:
        _need(ent, UNIVERSE_FEATURE[universe], {
            "sp500": "Your plan doesn't include S&P 500 scans.",
            "nasdaq": "NASDAQ scans are part of Pro.",
            "combo": "Combo scans are part of Pro.",
            "us_market": "US market scans are part of Premium.",
        }[universe])
        if universe == "combo":
            _need(ent, "can_scan_sp500", "Combo scans are part of Pro.")
    if universe == "ticker" and not req.get("ticker"):
        raise ValueError("ticker is required for a ticker scan.")
    if universe == "watchlist" and not req.get("watchlist_id"):
        raise ValueError("watchlist_id is required for a watchlist scan.")

    top_n = f.get("top_n")
    top_n = min(25, max_results) if top_n is None else int(top_n)
    if top_n > max_results:
        raise PlanError(f"Your plan shows up to {max_results} results per scan.")

    session = f.get("session") or "regular"
    if session == "premarket":
        _need(ent, "can_premarket", "Pre-market scans are part of Pro.")
    if session == "afterhours":
        _need(ent, "can_afterhours", "After-hours scans are part of Pro.")
    effective_session = session if session == "regular" or session == now_session else "regular"

    if f.get("unusual_volume"):
        _need(ent, "can_unusual_volume", "The unusual-volume filter is part of Pro.")
    if f.get("apply_gap_filter"):
        _need(ent, "can_scan_nasdaq", "The gap filter is part of Pro.")

    max_nasdaq, max_combo = f.get("max_nasdaq"), f.get("max_combo")
    if not full_lists:
        if max_nasdaq is not None and int(max_nasdaq) > PRO_NASDAQ_CAP:
            raise PlanError(f"Your plan scans up to {PRO_NASDAQ_CAP:,} NASDAQ tickers; Premium scans the full list.")
        if max_combo is not None and int(max_combo) > PRO_COMBO_CAP:
            raise PlanError(f"Your plan scans up to {PRO_COMBO_CAP:,} Combo tickers; Premium scans the full list.")

    min_price, max_price = float(f.get("min_price", 1.0)), float(f.get("max_price", 1000.0))
    if min_price > max_price:
        raise ValueError("min_price must not exceed max_price.")
    return {
        "universe": universe,
        "ticker": (req.get("ticker") or "").strip().upper() or None,
        "watchlist_id": req.get("watchlist_id"),
        "score_all": bool(req.get("score_all")),
        "profile": f.get("profile") or "regular",
        "session_requested": session,
        "session": effective_session,
        "min_price": min_price,
        "max_price": max_price,
        "min_dollar_vol": float(f.get("min_dollar_vol", 5_000_000.0)),
        "min_gap": float(f.get("min_gap", 1.0)),
        "apply_gap_filter": bool(f.get("apply_gap_filter")),
        "unusual_volume": bool(f.get("unusual_volume")),
        "top_n": top_n,
        "max_results": max_results,
        "full_lists": full_lists,
        "max_nasdaq": None if full_lists else int(max_nasdaq if max_nasdaq is not None else DEFAULT_NASDAQ),
        "max_combo": None if full_lists else int(max_combo if max_combo is not None else DEFAULT_COMBO),
        "early_breakout": bool(ent.get("can_early_breakout")),
        "is_admin": bool(is_admin),
        "tier": tier,
    }


# ---- running the scan -----------------------------------------------------------------------------
def _resolve_tickers(p: Dict[str, Any], username: str) -> List[str]:
    from api.scan_jobs import ScanFailed
    from ui.scan_providers import sanitize_universe_symbols

    if p["universe"] == "ticker":
        return [p["ticker"]]
    if p["universe"] == "watchlist":
        from api import user_data

        try:
            items = user_data.get_watchlist(username, int(p["watchlist_id"]))["items"]
        except user_data.NotFound as e:
            raise ScanFailed("That watchlist no longer exists.") from e
        tickers = sanitize_universe_symbols([i["ticker"] for i in items])
        if not tickers:
            raise ScanFailed("This watchlist has no tickers to scan.")
        return tickers

    from scan.universe_selection import resolve_scan_universe
    from ui.universe import apply_liquidity_filter_batch, filter_universe, load_nasdaq_universe, load_sp500_universe
    from ui.universe_us import US_MARKET_UNAVAILABLE, us_market_symbols

    market = MARKET[p["universe"]]
    state: Dict[str, Any] = {"min_dollar_vol": p["min_dollar_vol"]}
    if p["max_nasdaq"] is not None:
        state["max_nasdaq_scan"] = p["max_nasdaq"]
    if p["max_combo"] is not None:
        state["max_combo_scan"] = p["max_combo"]

    def trim(symbols):
        return list(apply_liquidity_filter_batch(list(symbols), min_price=p["min_price"],
                                                 min_avg_dollar_vol=p["min_dollar_vol"],
                                                 max_price=p["max_price"]))

    tickers = resolve_scan_universe(
        market, state, is_admin=p["is_admin"], safe_call=lambda fn, label=None: fn(),
        load_sp500_universe=load_sp500_universe, load_nasdaq_universe=load_nasdaq_universe,
        filter_universe=filter_universe, sanitize_symbols=sanitize_universe_symbols,
        label_suffix=market, combo_universe_transform=trim if market in ("COMBO", "US_MARKET") else None,
        load_us_market_universe=us_market_symbols, uncapped=p["full_lists"],
    )
    if not tickers:
        raise ScanFailed(US_MARKET_UNAVAILABLE if market == "US_MARKET"
                         else "The stock list isn't available right now. Try again in a few minutes.")
    return list(tickers)


def _label(p: Dict[str, Any], n: int) -> str:
    if p["universe"] == "ticker":
        return f"Search: {p['ticker']}"
    if p["universe"] == "watchlist":
        return f"Watchlist ({n} tickers)" + (" · all" if p["score_all"] else "")
    return LABELS[p["universe"]]


def run_scan(p: Dict[str, Any], username: str, report: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """The scan for job parameters `p`; returns the job result payload."""
    from api.scans import _scan_row, json_safe
    from scan.engine import run_breakout_scan
    from scan.execution import run_manual_scan_execution
    from ui.entitlement_view import redact_prebreakout_rows
    from ui.market_scans import top_setups
    from ui.scan_providers import apply_alpaca_extended_prices

    report({"phase": "loading_universe"})
    tickers = _resolve_tickers(p, username)
    label = _label(p, len(tickers))
    report({"phase": "scanning", "symbols": len(tickers)})

    bypass = p["universe"] == "watchlist" and p["score_all"]   # web: "score all" ignores the screens
    started = time.perf_counter()
    df = run_manual_scan_execution(
        runner=run_breakout_scan, tickers=tickers,
        premarket=p["session"] == "premarket", afterhours=p["session"] == "afterhours",
        unusual_volume=False if bypass else p["unusual_volume"],
        min_gap=0.0 if bypass else p["min_gap"],
        min_price=0.0 if bypass else p["min_price"],
        max_price=1_000_000.0 if bypass else p["max_price"],
        top_n=max(p["top_n"], len(tickers)) if bypass else p["top_n"],
        profile=p["profile"], apply_gap_filter=False if bypass else p["apply_gap_filter"],
        diagnostics=False, extended_price_transform=apply_alpaca_extended_prices,
        min_dollar_vol=0.0 if bypass else p["min_dollar_vol"],
    )
    duration = time.perf_counter() - started
    report({"phase": "finishing", "symbols": len(tickers)})

    _save_run(df, label, username, duration)
    opps = redact_prebreakout_rows(top_setups(df, n=100_000), allowed=p["early_breakout"])
    rows = [_scan_row(o) for o in opps]
    return json_safe({"label": label, "session": p["session"], "symbols_scanned": len(tickers),
                      "duration_s": round(duration, 1), "total": len(rows), "setups": rows})


def _save_run(df: Any, label: str, username: str, duration: float) -> None:
    """Scan history, as the web saves it (best effort; never fails the scan)."""
    try:
        from db.runs import save_run

        rows = 0 if df is None else len(df)
        save_run(f"{label} | {rows} results | {duration:.1f}s",
                 df.to_json(orient="records") if df is not None else "[]",
                 label=label, username=username, row_count=rows, duration_sec=duration, is_snapshot=False)
    except Exception:
        pass


def plan_or_raise(req: Dict[str, Any], account_ent: Dict[str, Any], now_session: Optional[str] = None) -> Dict[str, Any]:
    """plan_scan with the account's entitlements and the current market session."""
    from api.scans import max_results_for

    if now_session is None:
        from ui.app_runtime import get_market_session

        now_session = get_market_session()
    tier = account_ent["tier"]
    return plan_scan(req, entitlements=account_ent["entitlements"], tier=tier,
                     is_admin=account_ent["is_admin"], max_results=max_results_for(tier),
                     now_session=now_session)
