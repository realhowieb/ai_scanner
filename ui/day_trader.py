"""⚡ Day Trader live monitor: gappers, VWAP, RVOL, today's move.

Built on chunked Alpaca snapshots (cached ~30s). Beyond the table it now:
- shows the market state (premarket / open / after-hours / closed) honestly,
- adds an AH % column after the close (last vs today's close),
- highlights rows that are "in play" (big move or elevated RVOL),
- watches with you: optional toasts when a symbol moves ±X% while open,
- offers row actions (chart / trade plan / alert-me) without leaving the page,
- sources symbols from presets (watchlist / today's scan picks / mega-caps / custom),
- refreshes via st.fragment so only the table re-renders on each tick.

Pure helpers (market_state, detect_moves) are streamlit-free and unit-tested.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ui.safe_errors import show_error as _show_error

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be absent in minimal envs
    pd = None  # type: ignore[assignment]

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs; pure helpers still work
    st = None  # type: ignore[assignment]

MAX_SYMBOLS = 150
MEGA_CAPS = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, SPY, QQQ, IWM, DIA"
HIGHLIGHT_CHG_PCT = 3.0
HIGHLIGHT_RVOL = 2.0
DAY_TRADER_TABLE_COLUMNS = [
    "Ticker",
    "Open",
    "Last",
    "Change $",
    "Gap %",
    "ADX",
    "VWAP",
    "vs VWAP",
    "RVOL",
    "Volume (M)",
    "SuperTrend (13,2)",
    "EWO",
    "Direction",
    "DT Score",
    "Setup",
]


# ------------------------------ pure helpers -------------------------------


def market_state(
    now_utc: Optional[dt.datetime] = None,
    clock_is_open: Optional[bool] = None,
) -> str:
    """'premarket' | 'open' | 'afterhours' | 'closed' (US equities, ET).

    `clock_is_open` is the exchange's own answer (Alpaca /v2/clock) and, when
    provided, overrides the time-based guess for the regular session — the
    time-only logic is wrong on holidays and early-close days. Time buckets
    still classify the extended-hours windows.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    if clock_is_open is True:
        return "open"
    if et.weekday() >= 5:
        return "closed"
    minutes = et.hour * 60 + et.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        # Would be regular hours by the calendar — but the exchange says
        # closed (holiday) or already closed (early-close day).
        return "closed" if clock_is_open is False else "open"
    if 16 * 60 <= minutes < 20 * 60:
        return "afterhours"
    return "closed"


def _fetch_clock_is_open() -> Optional[bool]:
    """Ask Alpaca's trading clock whether the regular session is open."""
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_headers

        cfg = get_alpaca_config()
        headers = get_alpaca_headers()
        if not cfg or not headers:
            return None
        resp = requests.get(f"{cfg['base_url']}/v2/clock", headers=headers, timeout=5)
        if resp.status_code != 200:
            return None
        return bool((resp.json() or {}).get("is_open"))
    except Exception:
        return None


if st is not None:  # cached wrapper (the clock changes at minute granularity)

    @st.cache_data(ttl=60, show_spinner=False)
    def _clock_is_open() -> Optional[bool]:
        return _fetch_clock_is_open()

else:  # pragma: no cover
    _clock_is_open = _fetch_clock_is_open


def detect_moves(
    baseline: Dict[str, float], current: Dict[str, float], threshold_pct: float
) -> List[tuple]:
    """[(sym, pct_move_since_baseline)] for symbols past the threshold."""
    out: List[tuple] = []
    for sym, last in current.items():
        base = baseline.get(sym)
        if not base or not last:
            continue
        move = (last - base) / base * 100.0
        if abs(move) >= float(threshold_pct):
            out.append((sym, round(move, 2)))
    return out


_SYMBOL_RE = None


def after_hours_pct(last: Optional[float], close_today: Optional[float]) -> Optional[float]:
    """Extended-hours move: last trade vs the completed regular-session close.

    On Alpaca's IEX feed the current-day bar stops updating at the regular
    close, so after hours `dailyBar.c` is the official close and latestTrade
    carries extended-hours prints. Returns None unless both prices exist and
    genuinely differ (equal prices mean no AH trade yet — show nothing rather
    than a fake 0.00%).
    """
    try:
        if not last or not close_today:
            return None
        if float(last) == float(close_today):
            return None
        return round((float(last) / float(close_today) - 1.0) * 100.0, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _ema_cross_display(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "golden":
        return "Golden Cross"
    if text == "death":
        return "Death Cross"
    return "—"


def format_change_dollar(value: object) -> str:
    """Format an intraday dollar move as +$0.00 / -$0.00."""
    try:
        v = float(value)
        if v != v:
            return "—"
        return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def format_volume_millions(value: object) -> str:
    """Format raw share volume as millions for compact display."""
    try:
        v = float(value)
        if v != v:
            return "—"
        return f"{v / 1_000_000.0:,.2f}M"
    except (TypeError, ValueError):
        return "—"


def format_vwap_distance(value: object) -> str:
    """Format VWAP distance with explicit above/below context."""
    try:
        v = float(value)
        if v != v:
            return "—"
        if v > 0:
            return "above"
        if v < 0:
            return "below"
        return "at VWAP"
    except (TypeError, ValueError):
        return "—"


def format_supertrend_direction(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in ("green", "bullish", "up", "true"):
        return "🟢 Green"
    if text in ("red", "bearish", "down", "false"):
        return "🔴 Red"
    return "—"


def _ensure_day_trader_table_columns(df):
    """Return the canonical Day Trader table shape, filling missing fields."""
    return _ensure_day_trader_table_columns_for(df, DAY_TRADER_TABLE_COLUMNS)


def _ensure_day_trader_table_columns_for(df, columns):
    """Return a requested Day Trader table shape, filling missing fields."""
    if pd is None:
        return df
    out = df.copy()
    text_cols = {"Ticker", "SuperTrend (13,2)"}
    for col in columns:
        if col not in out.columns:
            out[col] = "—" if col in text_cols else float("nan")
    if "SuperTrend (13,2)" in out.columns:
        out["SuperTrend (13,2)"] = out["SuperTrend (13,2)"].apply(
            lambda v: "—" if pd.isna(v) else v
        )
    return out[columns]


def _market_feed_label(rows: List[Dict[str, Any]]) -> str:
    feeds = {
        str(r.get("market_data_feed") or "").strip().lower()
        for r in rows or []
        if r.get("market_data_feed")
    }
    if feeds == {"sip"}:
        return "Alpaca SIP"
    if feeds == {"iex"}:
        return "Alpaca IEX"
    if feeds:
        return "Alpaca mixed feeds"
    return "Alpaca"


def _volume_column_label(rows: List[Dict[str, Any]]) -> str:
    sources = {
        str(r.get("volume_source") or "").strip().lower()
        for r in rows or []
        if r.get("volume_source")
    }
    if any(src.endswith("_iex") or src == "alpaca_iex" for src in sources):
        return "Volume (IEX M)"
    return "Volume (M)"


def _table_columns_for_volume(volume_col: str) -> List[str]:
    return [volume_col if c == "Volume (M)" else c for c in DAY_TRADER_TABLE_COLUMNS]


def _parse_symbols(raw: str, max_symbols: int = 200) -> List[str]:
    """Parse, validate, and dedupe a comma-separated ticker list.

    Accepts 1-8 chars of A-Z / digits / dot / dash (covers class shares like
    BRK.B and BRK-B); silently drops anything else rather than sending junk to
    the quotes API.
    """
    global _SYMBOL_RE
    if _SYMBOL_RE is None:
        import re

        _SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,7}$")
    parts = [p.strip().upper() for p in (raw or "").replace("\n", ",").split(",")]
    out = [p for p in dict.fromkeys(parts) if p and _SYMBOL_RE.match(p)]
    return out[:max_symbols]


# ----------------------------- symbol sources ------------------------------


def _default_symbols(watch_tickers: List[str] | None) -> str:
    if watch_tickers:
        return ", ".join(dict.fromkeys(t.upper() for t in watch_tickers if str(t).strip()))
    return MEGA_CAPS


def _scan_pick_symbols(limit: int = 40) -> List[str]:
    """Tickers from the latest daily snapshot (already ranked model-first)."""
    try:
        from db.runs import list_snapshot_runs, load_many_run_results
        from ui.app_runtime import normalize_results_to_df

        runs = list_snapshot_runs(days=5, limit=5) or []
        if not runs:
            return []
        payloads = load_many_run_results([runs[0]["id"]])
        raw = payloads.get(int(runs[0]["id"]))
        df = normalize_results_to_df(raw) if raw else None
        if df is None or len(df) == 0:
            return []
        col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
        if not col:
            return []
        return [str(t).upper() for t in df[col].head(limit).tolist() if str(t).strip()]
    except Exception:
        return []


def _session_scan_symbols(label: str, limit: int = 40) -> List[str]:
    """Tickers from the most recent headless scan for a session ('premarket' /
    'postmarket'), so the monitor can watch that session's picks live.

    These runs are produced by the scheduler (scan.pre_post) and stored with the
    session as their run label. Returns [] when none is available yet.
    """
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df

        runs = list_runs(limit=60, include_snapshots=False) or []
        target = str(label).strip().lower()
        run = next(
            (r for r in runs if str(r.get("label") or "").strip().lower() == target),
            None,
        )
        if not run:
            return []
        raw = load_run_results(int(run["id"]))
        df = normalize_results_to_df(raw) if raw else None
        if df is None or len(df) == 0:
            return []
        col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
        if not col:
            return []
        return [str(t).upper() for t in df[col].head(limit).tolist() if str(t).strip()]
    except Exception:
        return []


if st is not None:

    @st.cache_data(ttl=900, show_spinner=False)
    def _scan_picks_cached() -> List[str]:
        return _scan_pick_symbols()

    @st.cache_data(ttl=300, show_spinner=False)
    def _session_scan_cached(label: str) -> List[str]:
        return _session_scan_symbols(label)

else:  # pragma: no cover - headless fallback
    _scan_picks_cached = _scan_pick_symbols
    _session_scan_cached = _session_scan_symbols


def day_trade_score(row: Dict[str, Any]) -> float:
    """Pure-intraday day-trade momentum score — no 20-day factors.

    Leads with today's move + VWAP alignment (intraday momentum), with the gap
    and relative volume as supporting signals, so it surfaces names moving *now*
    regardless of where they sit in a 20-day range. Reads the intraday fields
    build_day_trader_metrics already computes.
    """
    def _n(v) -> float:
        try:
            f = float(v)
            return f if f == f else 0.0  # drop NaN
        except (TypeError, ValueError):
            return 0.0

    chg = _n(row.get("chg_pct"))
    gap = _n(row.get("gap_pct"))
    rvol = _n(row.get("rvol"))
    vsvwap = _n(row.get("vs_vwap_pct"))

    score = abs(chg) * 2.0                      # intraday momentum (primary)
    aligned = (chg >= 0) == (vsvwap >= 0)       # on the right side of VWAP for the move
    score += (abs(vsvwap) if aligned else -abs(vsvwap)) * 1.0  # VWAP alignment
    score += abs(gap) * 0.8                     # gap (support)
    score += min(max(rvol, 0.0), 5.0) * 1.5     # volume surge (support, capped)
    return round(score, 3)


def _dedupe_syms(raw) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for sym in raw or []:
        u = str(sym or "").strip().upper()
        if u and not u.startswith("#") and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _sp500_universe() -> List[str]:
    """S&P 500 tickers (deduped)."""
    try:
        from scan.pre_post import _load_sp500

        return _dedupe_syms(_load_sp500() or [])
    except Exception:
        return []


def _nasdaq_universe() -> List[str]:
    """NASDAQ-listed tickers from the bundled nasdaq.txt (deduped)."""
    from pathlib import Path

    try:
        f = Path(__file__).resolve().parents[1] / "nasdaq.txt"
        if f.exists():
            return _dedupe_syms(f.read_text(encoding="utf-8").splitlines())
    except Exception:
        pass
    return []


def _movers_universe() -> List[str]:
    """S&P 500 + NASDAQ tickers (deduped), for the top-movers screen."""
    return _dedupe_syms([*_sp500_universe(), *_nasdaq_universe()])


_MOVERS_MIN_DOLLAR_VOL = 1_000_000  # skip illiquid micro-caps you can't day-trade
_MOVERS_MAX_STALE_DAYS = 6          # drop delisted/halted names (stale last trade)


def _is_stale(trade_ts, max_days: int = _MOVERS_MAX_STALE_DAYS) -> bool:
    """True when a snapshot's latest trade is older than ``max_days`` — the
    signature of a delisted/halted ticker (e.g. QMMM leaking in from a static
    universe file). Unknown/unparseable timestamps are treated as NOT stale so
    we never over-filter a live name on a data quirk.
    """
    if not trade_ts:
        return False
    try:
        import datetime as _dt

        d = _dt.datetime.fromisoformat(str(trade_ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - d).days > max_days
    except Exception:
        return False


def _top_movers_symbols(limit: int = 40, universe: List[str] | None = None) -> List[str]:
    """Names ranked by intraday day-trade momentum (no 20-day), over ``universe``
    (defaults to S&P 500 + NASDAQ).

    with_rvol=False on purpose — the universe can span the full NASDAQ composite
    (thousands of names), and RVOL needs a per-symbol daily-bar fetch that would
    be far too heavy here. The snapshot alone gives gap/change/VWAP (the momentum
    signals we lead with); a dollar-volume floor drops illiquid junk.
    """
    try:
        from market_data import build_day_trader_metrics

        universe = universe if universe is not None else _movers_universe()
        if not universe:
            return []
        rows = build_day_trader_metrics(universe, with_rvol=False) or []
        scored: List[tuple] = []
        for r in rows:
            t = r.get("ticker")
            if not t:
                continue
            if _is_stale(r.get("trade_ts")):
                continue  # delisted / halted — last trade too old to be a "mover"
            try:
                dvol = float(r.get("last") or 0) * float(r.get("volume") or 0)
            except (TypeError, ValueError):
                dvol = 0.0
            if dvol < _MOVERS_MIN_DOLLAR_VOL:
                continue
            s = day_trade_score(r)
            if s > 0:  # require real intraday activity
                scored.append((s, str(t).upper()))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set = set()
        out: List[str] = []
        for _s, t in scored:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


if st is not None:

    @st.cache_data(ttl=120, show_spinner="Screening S&P 500 + NASDAQ for today's movers…")
    def _top_movers_cached() -> List[str]:
        return _top_movers_symbols()

    @st.cache_data(ttl=120, show_spinner="Screening S&P 500 for today's movers…")
    def _sp500_movers_cached() -> List[str]:
        return _top_movers_symbols(universe=_sp500_universe())

    @st.cache_data(ttl=120, show_spinner="Screening NASDAQ for today's movers…")
    def _nasdaq_movers_cached() -> List[str]:
        return _top_movers_symbols(universe=_nasdaq_universe())

else:  # pragma: no cover - headless fallback
    _top_movers_cached = _top_movers_symbols
    _sp500_movers_cached = lambda: _top_movers_symbols(universe=_sp500_universe())  # noqa: E731
    _nasdaq_movers_cached = lambda: _top_movers_symbols(universe=_nasdaq_universe())  # noqa: E731


def _resolve_symbols(source: str, watch_tickers: List[str] | None) -> List[str]:
    if source == "🔥 Top movers (S&P 500 + NASDAQ)":
        movers = _top_movers_cached()
        if movers:
            return movers
        st.caption("Couldn't screen movers right now — using your watchlist.")
        return _parse_symbols(_default_symbols(watch_tickers))
    if source == "🔥 Top movers (S&P 500)":
        movers = _sp500_movers_cached()
        if movers:
            return movers
        st.caption("Couldn't screen S&P 500 movers right now — using your watchlist.")
        return _parse_symbols(_default_symbols(watch_tickers))
    if source == "🔥 Top movers (NASDAQ)":
        movers = _nasdaq_movers_cached()
        if movers:
            return movers
        st.caption("Couldn't screen NASDAQ movers right now — using your watchlist.")
        return _parse_symbols(_default_symbols(watch_tickers))
    if source in ("🌅 Premarket movers", "🌙 Postmarket movers"):
        label = "premarket" if source.startswith("🌅") else "postmarket"
        picks = _session_scan_cached(label)
        if picks:
            return picks
        st.caption(f"No recent {label} scan found — using your watchlist.")
        return _parse_symbols(_default_symbols(watch_tickers))
    if source == "Today's scan picks":
        picks = _scan_picks_cached()
        if picks:
            return picks
        st.caption("No recent scan snapshot found — using your watchlist.")
        return _parse_symbols(_default_symbols(watch_tickers))
    if source == "Mega-caps":
        return _parse_symbols(MEGA_CAPS)
    if source == "Custom":
        raw = st.text_input(
            "Symbols",
            value=st.session_state.get("dt_symbols") or _default_symbols(watch_tickers),
            key="dt_symbols",
            help="Comma-separated tickers.",
        )
        return _parse_symbols(raw)
    # "⭐ Watchlist" (and any fallback): use the resolved watchlist tickers.
    return _parse_symbols(_default_symbols(watch_tickers))


def _watchlist_source_tickers(fallback: List[str] | None) -> List[str] | None:
    """Render a watchlist picker and return the chosen list's tickers for the
    Day Trader source. Default selection = the user's is_default watchlist. An
    explicit 'All watchlists' merges them; otherwise a single list is used. Reads
    only the watchlist data layer — never a scan. Falls back to `fallback` when
    unauthenticated / no watchlists / unavailable."""
    username = (st.session_state.get("username") or "").strip().lower()
    if not username:
        return fallback
    try:
        from db.watchlists import get_watchlist_tickers, list_watchlists
    except Exception:
        return fallback
    wls = list_watchlists(username) or []
    if not wls:
        st.caption("No watchlists yet — create one on the My Watchlist page.")
        return fallback
    labels: List[str] = []
    id_by_label: Dict[str, int] = {}
    default_label = None
    for w in wls:
        lbl = w["name"] + (" (default)" if w.get("is_default") else "")
        labels.append(lbl)
        id_by_label[lbl] = w["id"]
        if w.get("is_default"):
            default_label = lbl
    labels.append("All watchlists")
    idx = labels.index(default_label) if default_label else 0
    choice = st.selectbox("Watchlist", labels, index=idx, key="dt_wl_source")
    if choice == "All watchlists":
        seen, out = set(), []
        for w in wls:
            for t in (get_watchlist_tickers(w["id"], username) or []):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out
    return get_watchlist_tickers(id_by_label[choice], username) or []


# --------------------------------- panel -----------------------------------


def render_day_trader_panel(
    watch_tickers: List[str] | None = None,
    *,
    max_symbols: int = MAX_SYMBOLS,
) -> None:
    """Render the live day-trader monitor."""
    if st is None:
        return
    try:
        from config import DAY_TRADER_ENABLED
    except Exception as e:
        print(f"[day_trader] config import failed — panel hidden: {type(e).__name__}: {e}")
        return
    if not DAY_TRADER_ENABLED:
        return
    from ui.showcase import screenshot_mode

    showcase = screenshot_mode()
    st.markdown("## ⚡ Day Trader — live")
    try:
        from ui.ticker_strip import render_ticker_strip

        # The "Watchlist" strip shows watchlist symbols, so drive its % changes
        # from the watchlist's own quotes — NOT the day-trader table (dt_rows),
        # whose symbols depend on the selected source (Mega-caps, Top movers, …)
        # and would leave non-overlapping watchlist tickers blank and reshuffle
        # which chips light up on every table refresh. The table rows are layered
        # on top only as a fresher quote for tickers that appear in both.
        render_ticker_strip(
            watch_tickers,
            label="Watchlist",
            quote_rows=(
                list(st.session_state.get("active_watchlist_quote_rows") or [])
                + list(st.session_state.get("dt_rows") or [])
            ),
        )
    except Exception:
        pass
    _render_state_banner()

    source_options = [
        "⭐ Watchlist",
        "🔥 Top movers (S&P 500 + NASDAQ)",
        "🔥 Top movers (S&P 500)",
        "🔥 Top movers (NASDAQ)",
        "🌅 Premarket movers",
        "🌙 Postmarket movers",
        "Today's scan picks",
        "Mega-caps",
        "Custom",
    ]
    if showcase:
        source = st.selectbox(
            "Symbols source", source_options,
            index=0 if watch_tickers else source_options.index("Mega-caps"), key="dt_source")
        refresh_label = "Off"
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            source = st.selectbox(
                "Symbols source",
                source_options,
                index=0 if watch_tickers else source_options.index("Mega-caps"),
                key="dt_source",
            )
        with c2:
            refresh_label = st.selectbox(
                "Auto-refresh", ["Off", "15s", "30s", "60s"], index=0, key="dt_refresh"
            )
        with c3:
            st.write("")
            if st.button("🔄 Refresh now", key="dt_refresh_btn"):
                # Clear only this page's cached fetches — st.cache_data.clear()
                # would nuke every app cache (models, history, quotes) and recreate
                # the slowness the caching work eliminated.
                for fn in (
                    "fetch_alpaca_snapshots",
                    "fetch_avg_daily_volume",
                    "fetch_ema_crosses",
                    "fetch_daily_range_metrics",
                ):
                    try:
                        import market_data

                        getattr(market_data, fn).clear()
                    except Exception:
                        pass
                try:
                    _scan_picks_cached.clear()  # type: ignore[attr-defined]
                except Exception:
                    pass
                st.rerun()

    # ⭐ Watchlist source: let the user pick WHICH of their watchlists drives the
    # symbols (default = their is_default list). Selecting a watchlist just feeds
    # its tickers into the metrics pipeline — it never runs a market scan.
    effective_watch = watch_tickers
    if source == "⭐ Watchlist":
        effective_watch = _watchlist_source_tickers(watch_tickers)
    symbols = _resolve_symbols(source, effective_watch)
    if not symbols:
        st.info("No symbols to monitor for this source.")
        return
    if len(symbols) > max_symbols:
        st.caption(f"Showing the first {max_symbols} of {len(symbols)} symbols.")
        symbols = symbols[:max_symbols]

    # Watch-mode movement notifications (session-local; compares each refresh
    # to the price when you started watching, resets per symbol after firing).
    if showcase:
        notify = False
        move_thr = 2.0
    else:
        w1, w2 = st.columns([2, 1])
        notify = w1.checkbox(
            "🔔 Notify me on big moves while watching",
            key="dt_notify",
            help=(
                "This browser tab only, while it stays open — baselines reset on "
                "refresh/logout. For alerts that persist and email you, use the "
                "Alerts page."
            ),
        )
        if notify:
            w1.caption("⏱️ Session-only — for persistent alerts use the 🔔 Alerts page.")
        move_thr = w2.number_input(
            "Move ≥ %", min_value=0.5, value=2.0, step=0.5, key="dt_notify_thr",
            disabled=not notify,
        )
    # Reset baselines when the watched symbol set changes, so stale entries
    # from a previous source can't produce confusing move calculations.
    _sym_sig = ",".join(sorted(symbols))
    if st.session_state.get("dt_watch_symbols") != _sym_sig:
        st.session_state["dt_watch_symbols"] = _sym_sig
        st.session_state.pop("dt_watch_baseline", None)

    interval_s = {"Off": 0, "15s": 15, "30s": 30, "60s": 60}.get(refresh_label, 0)

    show_score = source.startswith("🔥 Top movers")

    def _body() -> None:
        _render_table(symbols, notify=notify, move_thr=float(move_thr), show_score=show_score)

    if interval_s and hasattr(st, "fragment"):
        # Fragment re-renders only the table on each tick — the rest of the
        # page (and the wider app) doesn't rerun.
        st.fragment(run_every=f"{interval_s}s")(_body)()
    else:
        if interval_s:
            try:
                from streamlit_autorefresh import st_autorefresh

                st_autorefresh(interval=interval_s * 1000, key="dt_autorefresh")
            except Exception:
                st.caption("Auto-refresh unavailable; use 🔄 Refresh now.")
        _body()

    _render_row_actions()


def _render_state_banner() -> None:
    state = market_state(clock_is_open=_clock_is_open())
    banner = {
        "premarket": "🌅 **Premarket** — extended-hours trades shown; volume is thin.",
        "open": "🔔 **Market open** — live regular-session data.",
        "afterhours": "🌙 **After-hours** — extended-hours trades shown; see the AH % column.",
        "closed": "💤 **Market closed** — showing the last session's data.",
    }.get(state, "")
    if banner:
        st.caption(banner)


def _render_table(
    symbols: List[str], *, notify: bool, move_thr: float, show_score: bool = False
) -> None:
    from market_data import build_day_trader_metrics

    try:
        rows = build_day_trader_metrics(symbols)
    except Exception as e:
        _show_error("live data", e, level="info")
        return
    if not rows:
        st.caption("No live data (market closed, Alpaca not configured, or symbols not found).")
        return
    if len(rows) < len(symbols):
        missing = len(symbols) - len(rows)
        st.caption(
            f"⚠️ Showing {len(rows)} of {len(symbols)} symbols — {missing} "
            "returned no quote (unknown ticker or a partial data fetch)."
        )

    state = market_state(clock_is_open=_clock_is_open())

    # Movement watch: toast symbols that moved past the threshold since the
    # baseline (price at watch start / last notification).
    moved_now: set = set()
    if notify:
        current = {r["ticker"]: r["last"] for r in rows if r.get("last")}
        baseline = st.session_state.get("dt_watch_baseline") or {}
        if baseline:
            for sym, move in detect_moves(baseline, current, move_thr):
                moved_now.add(sym)
                try:
                    st.toast(f"⚡ {sym} {move:+.1f}% since you started watching")
                except Exception:
                    pass
                baseline[sym] = current[sym]  # reset so it doesn't re-toast every tick
        # First render (or newly appeared symbols): set baselines silently.
        for sym, px in current.items():
            baseline.setdefault(sym, px)
        st.session_state["dt_watch_baseline"] = baseline
    else:
        st.session_state.pop("dt_watch_baseline", None)

    df = pd.DataFrame(rows).rename(
        columns={
            "ticker": "Ticker",
            "open": "Open",
            "last": "Last",
            "change_dollar": "Change $",
            "chg_pct": "Chg %",
            "gap_pct": "Gap %",
            "adx": "ADX",
            "vwap": "VWAP",
            "vs_vwap_pct": "vs VWAP",
            "volume": "Volume",
            "rvol": "RVOL",
            "ema_cross": "EMA Cross",
            "atr_pct": "ATR %",
            "donchian_pos": "Range %",
            "bb_pctb": "%B",
            "supertrend_direction": "SuperTrend (13,2)",
            "ewo": "EWO",
        }
    )
    # Donchian breakout + Bollinger squeeze as pre-formatted string columns
    # (icons/em-dash), so a None cell never renders as the literal "None".
    if "donchian_breakout" in df.columns:
        df["20d B/O"] = df["donchian_breakout"].apply(
            lambda v: "🔼 high" if v == "up" else ("🔽 low" if v == "down" else "—")
        )
    if "bb_squeeze" in df.columns:
        df["Squeeze"] = df["bb_squeeze"].apply(lambda v: "🎯" if bool(v) else "—")
    # Run 32 — deterministic signal intelligence (Direction / DT Score / Setup),
    # synthesized locally from the same raw metric rows (no scan, no network).
    # Rows lacking ADX/SuperTrend/EWO show "—" honestly (insufficient evidence).
    try:
        from analytics.day_trade_intel import day_trade_intelligence, direction_icon

        intel = [day_trade_intelligence(r) for r in rows]
        df["Direction"] = [
            f"{direction_icon(i['direction'])} {i['direction'].title()}"
            if i["quality"] != "insufficient" else "—"
            for i in intel
        ]
        df["DT Score"] = [i["score"] if i["score"] is not None else None for i in intel]
        df["Setup"] = [
            i["quality"].title() if i["quality"] != "insufficient" else "—" for i in intel
        ]
    except Exception:
        pass
    volume_col = _volume_column_label(rows)
    if "Volume" in df.columns:
        df[volume_col] = pd.to_numeric(df["Volume"], errors="coerce") / 1_000_000.0
    if "SuperTrend (13,2)" in df.columns:
        df["SuperTrend (13,2)"] = df["SuperTrend (13,2)"].apply(format_supertrend_direction)
    if "EMA Cross" in df.columns:
        df["EMA Cross"] = df["EMA Cross"].apply(_ema_cross_display)
    # After-hours change: last trade vs today's official close. Rendered as a
    # pre-formatted STRING column ("—" when missing). Streamlit's grid shows a
    # NaN/None cell as the literal "None" regardless of the styler's formatter,
    # so the only reliable way to get an em-dash is to put the display string in
    # the data itself. Colored below by parsing the sign.
    if state in ("afterhours", "closed") and "close_today" in df.columns:
        df["AH %"] = [
            "—" if (v := after_hours_pct(r.get("Last"), r.get("close_today"))) is None
            else f"{v:+.2f}%"
            for r in df.to_dict(orient="records")
        ]

    df = _ensure_day_trader_table_columns_for(df, _table_columns_for_volume(volume_col))
    from ui.showcase import DAY_TRADER_SHOWCASE_COLUMNS, screenshot_mode, select_columns

    if screenshot_mode():
        df = select_columns(df, DAY_TRADER_SHOWCASE_COLUMNS)

    # Pin Ticker so it stays put while the rest scrolls; degrade gracefully on
    # older Streamlit that lacks column_config/pinned.
    try:
        col_cfg = {"Ticker": st.column_config.Column(width="small", pinned=True)}
        st.dataframe(_styled(df, moved_now), hide_index=True, width="stretch",
                     column_config=col_cfg)
    except Exception:
        st.dataframe(_styled(df, moved_now), hide_index=True, width="stretch")
    if not screenshot_mode():
        st.caption("↔ Swipe the table sideways on mobile.")
    if not any(
        r.get("adx") is not None or r.get("supertrend_direction") is not None or r.get("ewo") is not None
        for r in rows
    ):
        st.caption(
            "ADX, SuperTrend and EWO require cached daily OHLC history. "
            "They show — until historical bars are available; Refresh now retries the daily-bar fetch."
        )

    # Stash for the row-action picker rendered outside the fragment.
    st.session_state["dt_rows"] = rows

    now = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
    feed_label = _market_feed_label(rows)
    rvol_caption = (
        "RVOL = today's IEX volume ÷ 20-day IEX avg"
        if volume_col == "Volume (IEX M)"
        else "RVOL = today's volume ÷ 20-day avg"
    )
    st.caption(
        f"As of {now} · Open = regular-session open · Gap % vs prior close · "
        f"{rvol_caption} · Market data: {feed_label} · "
        "Highlight = ±3% move or 2× RVOL · verify before trading."
    )


def _styled(df, moved_now: set):
    def _missing(val) -> bool:
        try:
            return val is None or bool(pd.isna(val))
        except Exception:
            return val is None

    def color_pct(val):
        if _missing(val):
            return ""
        try:
            return "color: #16a34a" if float(val) >= 0 else "color: #dc2626"
        except (TypeError, ValueError):
            return ""

    def color_pct_str(val):
        # AH % is a pre-formatted string ("+1.2%" / "-0.5%" / "—"); color by sign.
        if not isinstance(val, str):
            return ""
        if val.startswith("-"):
            return "color: #dc2626"
        if val.startswith("+"):
            return "color: #16a34a"
        return ""

    def _pct(v):
        return "—" if _missing(v) else f"{float(v):+.2f}%"

    def _vwap_distance(v):
        return format_vwap_distance(v)

    def _price(v):
        return "—" if _missing(v) else f"${float(v):,.2f}"

    def _signed_dollar(v):
        return format_change_dollar(v)

    def _signed_number(v):
        return "—" if _missing(v) else f"{float(v):+.2f}"

    def _rvol(v):
        return "—" if _missing(v) else f"{float(v):.2f}×"

    def _vol_m(v):
        return "—" if _missing(v) else f"{float(v):,.2f}M"

    def _one_decimal(v):
        return "—" if _missing(v) else f"{float(v):.1f}"

    def vwap_heat(val):
        # Diverging heat: green above VWAP, red below, intensity by magnitude,
        # neutral at 0 (midpoint stays uncolored per the diverging rule).
        if _missing(val):
            return ""
        try:
            value = float(val)
        except (TypeError, ValueError):
            return ""
        if value == 0:
            return ""
        alpha = min(abs(value) / 3.0, 1.0) * 0.35
        return (
            f"background-color: rgba(22, 163, 74, {alpha:.2f})"
            if value > 0
            else f"background-color: rgba(220, 38, 38, {alpha:.2f})"
        )

    def in_play(row):
        chg = row.get("Chg %")
        rvol = row.get("RVOL")
        hot = (chg is not None and not pd.isna(chg) and abs(chg) >= HIGHLIGHT_CHG_PCT) or (
            rvol is not None and not pd.isna(rvol) and rvol >= HIGHLIGHT_RVOL
        )
        moved = row.get("Ticker") in moved_now
        style = ""
        if moved:
            style = "background-color: rgba(250, 204, 21, 0.18)"
        elif hot:
            style = "background-color: rgba(59, 130, 246, 0.12)"
        return [style] * len(row)

    def _pct_pos(v):  # ATR % / Range % / %B — plain single-sided percent
        return "—" if _missing(v) else f"{float(v):.1f}%"

    fmt = {}
    for col in ("Chg %", "Gap %", "vs VWAP"):  # AH % is pre-formatted to strings
        if col in df.columns:
            fmt[col] = _pct
    if "vs VWAP" in df.columns:
        fmt["vs VWAP"] = _vwap_distance
    for col in ("ATR %", "Range %", "%B"):
        if col in df.columns:
            fmt[col] = _pct_pos
    for col in ("Open", "Last", "VWAP"):
        if col in df.columns:
            fmt[col] = _price
    if "Change $" in df.columns:
        fmt["Change $"] = _signed_dollar
    if "ADX" in df.columns:
        fmt["ADX"] = _one_decimal
    if "RVOL" in df.columns:
        fmt["RVOL"] = _rvol
    for col in ("Volume (M)", "Volume (IEX M)"):
        if col in df.columns:
            fmt[col] = _vol_m
    if "EWO" in df.columns:
        fmt["EWO"] = _signed_number

    try:
        styler = df.style.format(fmt).apply(in_play, axis=1)
        for col in ("Change $", "Chg %", "Gap %", "vs VWAP", "EWO"):
            if col in df.columns:
                styler = styler.map(color_pct, subset=[col])
        if "AH %" in df.columns:
            styler = styler.map(color_pct_str, subset=["AH %"])
        if "vs VWAP" in df.columns:
            styler = styler.map(vwap_heat, subset=["vs VWAP"])
        return styler
    except Exception:
        return df


def _watchlist_add_feedback(pick: str, wl_name: str, result: Dict[str, Any]) -> str:
    """Deterministic feedback string for a watchlist add (no duplicates)."""
    if result.get("added"):
        return f"{pick} added to {wl_name}."
    if result.get("already_present"):
        return f"{pick} is already in {wl_name}."
    return f"Couldn't add {pick} to {wl_name} right now."


def _render_watchlist_action(pick: str) -> None:
    """Add the picked Day Trader symbol to a persistent watchlist.

    Reuses the canonical watchlist data layer (db.watchlists) — no second
    watchlist implementation, no scan is triggered. Shows the default watchlist,
    existing membership, a quick 'add to default', a destination picker, and
    create-new. Adding here never removes the symbol from Day Trader."""
    username = (st.session_state.get("username") or "").strip().lower()
    if not username:
        st.caption("Sign in to add symbols to a watchlist.")
        return
    try:
        from db.watchlists import (
            add_tickers_to_watchlist,
            create_watchlist,
            get_watchlist_tickers,
            list_watchlists,
            set_default_watchlist,
        )
    except Exception:
        st.caption("Watchlists are unavailable right now.")
        return

    pick_u = str(pick).strip().upper()  # watchlist tickers are stored upper-cased
    wls = list_watchlists(username) or []
    # Persisted membership (which watchlists already contain the symbol).
    member_ids = set()
    for wl in wls:
        try:
            if pick_u in set(get_watchlist_tickers(wl["id"], username) or []):
                member_ids.add(wl["id"])
        except Exception:
            pass
    # Reflect this session's just-added ids immediately (no full refresh needed).
    just = st.session_state.setdefault("dt_wl_added", {})
    member_ids |= set(just.get(pick_u, set()))
    # Only a genuinely is_default watchlist is treated as the default — never an
    # arbitrary first list.
    default = next((w for w in wls if w.get("is_default")), None)

    st.markdown(f"**Add {pick_u} to a watchlist**")

    def _do_add(wid: int, name: str) -> None:
        res = add_tickers_to_watchlist(username, [pick_u], wid)
        just.setdefault(pick_u, set()).add(wid)
        member_ids.add(wid)
        st.success(_watchlist_add_feedback(pick_u, name, res))

    # Quick add to the default watchlist (fastest path) — only with a real default.
    if default:
        in_default = default["id"] in member_ids
        label = f"⭐ Add to {default['name']}" + (" ✓" if in_default else " (default)")
        if st.button(label, key="dt_wl_quickadd", disabled=in_default):
            _do_add(default["id"], default["name"])
    elif wls:
        st.caption("No default watchlist set — pick a destination below.")

    # Choose a destination (and set a default when none exists).
    if wls:
        labels = {f"{w['name']}" + (" (default)" if w.get("is_default") else ""): w["id"] for w in wls}
        choice = st.selectbox("Add to" if default else "Watchlist", list(labels.keys()), key="dt_wl_dest")
        wid = labels[choice]
        if st.button("Add", key="dt_wl_add", disabled=wid in member_ids):
            _do_add(wid, choice)
        if not default:
            if st.button(f"Set “{choice}” as default", key="dt_wl_setdefault"):
                if set_default_watchlist(wid, username):
                    st.success(f"{choice} is now your default watchlist.")

    # Create a new watchlist and add to it.
    with st.expander("➕ Create new watchlist"):
        new_name = st.text_input("Name", key="dt_wl_new_name", label_visibility="collapsed",
                                 placeholder="e.g. Momentum")
        if st.button("Create & add", key="dt_wl_create"):
            name = (new_name or "").strip()
            if not name:
                st.caption("Enter a name first.")
            else:
                # create_watchlist raises ValueError on a duplicate/invalid name.
                try:
                    wid = create_watchlist(username, name)
                except ValueError as e:
                    st.caption(str(e))
                    wid = None
                except Exception:
                    st.caption("Couldn't create that watchlist right now.")
                    wid = None
                if wid:
                    _do_add(wid, name)

    # Membership summary rendered LAST so an add in this run is reflected now.
    if member_ids:
        st.caption("✓ Already in: " + ", ".join(w["name"] for w in wls if w["id"] in member_ids))


def _render_symbol_intel(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact deterministic intelligence summary for the selected symbol.
    Returns the intel dict so callers (e.g. Trade plan) can reuse it."""
    try:
        from analytics.day_trade_intel import day_trade_intelligence, direction_icon
    except Exception:
        return {}
    intel = day_trade_intelligence(row)
    if intel.get("quality") == "insufficient":
        st.caption("⚪ Neutral · insufficient data for a reliable read.")
        return intel
    icon = direction_icon(intel["direction"])
    score = intel["score"]
    score_txt = "—" if score is None else f"{score:.0f}"
    st.markdown(f"{icon} **{intel['direction'].title()}** · DT Score {score_txt} "
                f"· **{intel['quality'].title()}**")
    if intel["reasons"]:
        st.caption(" · ".join(intel["reasons"]))
    for c in intel["conflicts"]:
        st.caption(f"⚠ {c}")
    return intel


def _render_row_actions() -> None:
    """Chart / trade plan / watchlist / alert for a picked ticker (outside the fragment)."""
    rows = st.session_state.get("dt_rows") or []
    if not rows:
        return
    tickers = [r["ticker"] for r in rows if r.get("ticker")]
    if not tickers:
        return
    st.markdown("**Act on a symbol**")
    # Keep the selection valid: if the previously picked symbol dropped out of the
    # current Day Trader results, reset so the selectbox falls to the first valid
    # ticker (never errors on a stale value). Symbols always come from dt_rows.
    if st.session_state.get("dt_action_ticker") not in tickers:
        st.session_state.pop("dt_action_ticker", None)
    pick = st.selectbox("Ticker", tickers, key="dt_action_ticker", label_visibility="collapsed")
    row = next((r for r in rows if r.get("ticker") == pick), None)
    if not row:
        return
    intel = _render_symbol_intel(row)  # compact Direction / DT Score / Setup / why
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("📈 Chart", key="dt_act_chart"):
        st.session_state["dt_show_chart"] = pick
    if a2.button("🎯 Trade plan", key="dt_act_plan"):
        st.session_state["dt_show_plan"] = pick
    if a3.button("⭐ Watchlist", key="dt_act_wl"):
        st.session_state["dt_show_wl"] = pick
    if a4.button("🔔 Alert", key="dt_act_alert"):
        st.session_state["alert_price_tk"] = pick
        if row.get("last"):
            st.session_state["alert_price_val"] = round(float(row["last"]), 2)
        try:
            st.switch_page("pages/alerts.py")
        except Exception:
            st.caption("Open the Alerts page from the sidebar — the form is pre-filled.")

    if st.session_state.get("dt_show_wl") == pick:
        _render_watchlist_action(pick)

    if st.session_state.get("dt_show_chart") == pick:
        try:
            from ui.charts import render_chart_for_ticker

            render_chart_for_ticker(pick, key=f"dt_chart_{pick}")
        except Exception:
            st.caption("Chart unavailable.")
    if st.session_state.get("dt_show_plan") == pick:
        try:
            from ui.trade_plan import render_trade_plan

            # Pass the deterministic Day Trader intelligence as context so the
            # trade plan can explain why the ticker surfaced (no engine rewrite).
            plan_row: Dict[str, Any] = {
                "Ticker": pick, "Last": row.get("last"),
                "dt_direction": intel.get("direction"),
                "dt_score": intel.get("score"),
                "dt_setup": intel.get("quality"),
                "dt_reasons": intel.get("reasons"),
                "dt_conflicts": intel.get("conflicts"),
            }
            if intel.get("quality") and intel["quality"] != "insufficient":
                st.caption(
                    f"Context: {intel['direction'].title()} · DT "
                    f"{'—' if intel.get('score') is None else round(intel['score'])} · "
                    f"{intel['quality'].title()}. Verify before trading.")
            render_trade_plan(plan_row, locked=False)
        except Exception:
            st.caption("Trade plan unavailable.")
