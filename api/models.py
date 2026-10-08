"""Response models for the HSF API (P1-59).

Every route declares one, so /docs documents each payload and iOS/Android/web
clients can be generated from the OpenAPI schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
    ok: bool


class Ready(BaseModel):
    ok: bool
    database: Literal["ok"]
    latest_scan_at: Optional[str] = Field(default=None, description="Latest market scan (ISO); null when none")
    scan_age_minutes: Optional[float] = None
    stale: Optional[bool] = Field(default=None, description="A scheduled full-market scan was missed (see Market.stale)")
    expected_scan_at: Optional[str] = Field(default=None, description="When stale: the missed scan slot (ISO)")


class TokenPair(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")
    refresh_token: str = Field(description="Single use; POST /v1/auth/refresh returns a new one")


class Me(BaseModel):
    email: str
    name: Optional[str] = None
    plan: Literal["basic", "pro", "premium", "admin"]
    plan_label: str = Field(description="Customer-facing name, e.g. Free, Pro")
    is_admin: bool
    alert_limit: int
    email_verified: bool = Field(default=True, description="False until the emailed link is used; needed to upgrade and for email alerts")
    entitlements: Dict[str, bool] = Field(description="Feature flags, e.g. can_day_trader, can_ai_notes")


class Market(BaseModel):
    phase: Literal["premarket", "open", "afterhours", "closed"]
    latest_scan_at: Optional[str] = Field(default=None, description="Latest full-market scan (ISO); null when none")
    stale: Optional[bool] = Field(default=None, description=(
        "True when a scheduled full-market scan was missed, so the data is older than the schedule "
        "promises; null when unknown. Overnight, weekends and holidays alone are never stale."))
    expected_scan_at: Optional[str] = Field(default=None, description="When stale: the missed scan slot (ISO)")


class Mover(BaseModel):
    ticker: str
    pct: Optional[float] = Field(default=None, description="Move in percent vs the previous close")
    last: Optional[float] = None
    score: Optional[int] = Field(default=None, description="HSF Score when the name qualifies")


class SessionCard(BaseModel):
    """Before the open (shown 8:35-10:30 ET) / After the close. Pro+; below Pro `locked`
    is true and `movers` empty."""
    scan_at: Optional[str] = Field(default=None, description="ISO time of the session scan")
    locked: bool
    movers: List[Mover] = []


class Setup(BaseModel):
    ticker: str
    score: int
    primary_setup: Optional[str] = None
    status: Optional[str] = None
    n_signals: Optional[int] = None
    last: Optional[float] = None
    chg_pct: Optional[float] = None
    gap_pct: Optional[float] = None
    rvol: Optional[float] = None
    prob: Optional[float] = Field(default=None, description="PreBreakout model output; null below Premium")


class TopSetups(BaseModel):
    state: Literal["qualifying", "no_qualifying", "empty_scan"]
    threshold: Optional[int] = None
    scan_at: Optional[str] = None
    setups: List[Setup] = []


class Standout(BaseModel):
    ticker: str
    score: int
    setup: Optional[str] = None


class Recap(BaseModel):
    day: str = Field(description="ET date of the recapped session (YYYY-MM-DD)")
    title: str
    scans: int = Field(description="Full-market scans that day")
    premarket_scans: int = 0
    postmarket_scans: int = 0
    entered: List[str] = Field(default=[], description="'TICKER (score)', HSF 40+, strongest first")
    left: List[str] = []
    standouts: List[Standout] = []


class SectionError(BaseModel):
    section: str
    error: str = Field(description="Error type only, never details")


class Today(BaseModel):
    as_of: str
    market: Market
    before_open: Optional[SessionCard] = None
    top_setups: Optional[TopSetups] = None
    after_close: Optional[SessionCard] = None
    recap: Optional[Recap] = None
    errors: List[SectionError] = []


# ---- step 5: scans and stock detail --------------------------------------------------------------
class ScanSetup(Setup):
    signals: List[str] = []
    fading: bool = False
    breakout_score: Optional[float] = None
    prob_rank: Optional[int] = Field(default=None, description=(
        "Premium. Where the raw PreBreakout model score sits among this scan's setups, as "
        "'top N%' (1 = strongest). Tells apart names that share the calibrated floor in prob."))


class LatestScan(BaseModel):
    scan_at: Optional[str] = Field(default=None, description="ISO time of the market scan; null when none exists")
    total: int = Field(description="Setups matching the filters, before the plan cap")
    max_results: int = Field(description="Rows this plan sees (Free 25, Pro 100, Premium 200)")
    limited: bool = Field(description="True when the plan cap hides some matching setups")
    stale: Optional[bool] = Field(default=None, description="A scheduled full-market scan was missed (see Market.stale)")
    setups: List[ScanSetup] = []


class LifecycleEvent(BaseModel):
    time: Optional[str] = None
    score: Optional[int] = None
    status: Optional[str] = None
    label: Optional[str] = None
    movement: Optional[str] = None
    delta: Optional[int] = None
    signals: List[str] = []
    signals_added: List[str] = []
    signals_removed: List[str] = []


class Bar(BaseModel):
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None


class WatchlistRef(BaseModel):
    id: int
    name: str


class Alert(BaseModel):
    id: int
    type: Literal["breakout", "watchlist", "price", "move", "rvol", "ema_cross", "ewo_cross"]
    ticker: Optional[str] = None
    threshold: Optional[float] = None
    direction: Optional[str] = None
    watchlist_only: bool = False
    enabled: bool
    last_fired_at: Optional[str] = None
    created_at: Optional[str] = None


class StockDetail(BaseModel):
    ticker: str
    scan_at: Optional[str] = Field(default=None, description="Latest market scan used")
    in_latest_scan: bool = Field(description="The ticker had at least one row in that scan")
    has_setup: bool = Field(description="It qualifies as an HSF setup (or has a recent recorded score)")
    from_history: bool = Field(description="Score comes from the last recorded observation, not the latest scan")
    price: Optional[float] = None
    change_pct: Optional[float] = None
    hsf_score: Optional[int] = None
    status: Optional[str] = None
    primary_setup: Optional[str] = None
    signals: List[str] = []
    score_components: Optional[Dict[str, float]] = None
    movement: Optional[str] = Field(default=None, description="RISING, FALLING, UNCHANGED, NEW, NO_BASELINE or VERSION_CHANGED")
    score_change: Optional[int] = None
    reasons: List[str] = []
    risks: List[str] = []
    watch_next: List[str] = []
    breakout_score: Optional[float] = None
    prob: Optional[float] = Field(default=None, description="PreBreakout model output; null below Premium")
    earnings_days: Optional[int] = None
    history_summary: Optional[Dict[str, Any]] = None
    historical_context: Optional[Dict[str, Any]] = Field(default=None, description="Matured outcomes for this score range")
    outcome_cohort: Optional[Dict[str, Any]] = None
    historical_locked: bool = Field(default=False, description="True below Pro: historical research "
                                    "(history_summary, historical_context, outcome_cohort) is a Pro feature")
    lifecycle: List[LifecycleEvent] = []
    bars: List[Bar] = Field(default=[], description="Daily bars cached by the scans, oldest first (up to 120)")
    bars_as_of: Optional[str] = None
    watchlists: List[WatchlistRef] = Field(default=[], description="Your watchlists that hold this ticker")
    alerts: List[Alert] = Field(default=[], description="Your alerts on this ticker")


# ---- step 6: watchlists and alerts ---------------------------------------------------------------
class Watchlist(BaseModel):
    id: int
    name: str
    is_default: bool
    symbol_count: int


class WatchlistItem(BaseModel):
    ticker: str
    added_at: Optional[str] = None
    price_when_added: Optional[float] = None
    note: Optional[str] = None


class WatchlistDetail(Watchlist):
    items: List[WatchlistItem] = []


class TickersResult(BaseModel):
    added: List[str] = []
    already_present: List[str] = []
    invalid: List[str] = Field(default=[], description="Not valid ticker symbols; nothing was saved for them")


class Alerts(BaseModel):
    limit: int = Field(description="Alerts this plan may have (Free 1, Pro 5, Premium 25)")
    used: int
    email_enabled: bool = Field(description="Pro+ get alert emails; everyone gets in-app alerts")
    alerts: List[Alert] = []


class AlertThreshold(BaseModel):
    min: float
    min_exclusive: bool = Field(description="True: the value must be greater than min (price); else at least min")
    max: float
    label: str
    default: Optional[float] = Field(default=None, description="The web form's starting value")


class AlertType(BaseModel):
    type: Literal["breakout", "watchlist", "price", "move", "rvol", "ema_cross", "ewo_cross"]
    label: str
    description: str
    needs_ticker: bool
    threshold: Optional[AlertThreshold] = Field(default=None, description="Null when the type takes no threshold")
    directions: List[str] = Field(default=[], description="Allowed `direction` values; empty when not used")
    watchlist_only_option: bool = Field(description="True when `watchlist_only` applies (breakout)")


class AlertEvent(BaseModel):
    id: int
    alert_id: Optional[int] = None
    ticker: Optional[str] = None
    message: str
    fired_at: Optional[str] = None


# ---- account step: sign-up, verification, passwords, preferences, billing -----------------------
class SignupResult(TokenPair):
    email: str
    verification_sent: bool = Field(description="False when email isn't configured; the account still works")


class Message(BaseModel):
    ok: bool = True
    message: str


class EmailPrefs(BaseModel):
    digest: bool = Field(description="Morning market digest")
    evening: bool = Field(description="Evening market wrap")
    alerts: bool = Field(description="Alert emails")


class BillingLink(BaseModel):
    url: str = Field(description="Stripe-hosted page to open in the browser")
    mode: Literal["checkout", "portal"]


class Device(BaseModel):
    id: int = Field(description="Use with DELETE /v1/me/devices/{id}")
    provider: Literal["apns", "fcm", "expo"]
    platform: Literal["ios", "android"]
    device_name: Optional[str] = None
    app_version: Optional[str] = None
    created_at: Optional[str] = None
    last_seen_at: Optional[str] = None


# ---- custom scans (POST /v1/scans) --------------------------------------------------------------
class ScanProgress(BaseModel):
    phase: Literal["queued", "starting", "loading_universe", "scanning", "finishing", "complete", "failed"]
    symbols: Optional[int] = Field(default=None, description="Stocks being scanned (after the liquidity pre-filter)")
    elapsed_s: Optional[float] = None


class ScanParams(BaseModel):
    """The parameters the scan actually used (after plan rules)."""
    universe: Literal["sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker"]
    ticker: Optional[str] = None
    watchlist_id: Optional[int] = None
    score_all: bool = False
    profile: Literal["regular", "aggressive", "conservative"]
    session_requested: Literal["regular", "premarket", "afterhours"]
    session: Literal["regular", "premarket", "afterhours"] = Field(
        description="Session scanned: pre-market / after-hours outside that session scan the regular session")
    min_price: float
    max_price: float
    min_dollar_vol: float
    min_gap: float
    apply_gap_filter: bool
    unusual_volume: bool
    top_n: int
    max_results: int = Field(description="This plan's row cap (Free 25, Pro 100, Premium 200)")
    full_lists: bool = Field(description="True for Premium/admin: NASDAQ and Combo scan the full lists")
    max_nasdaq: Optional[int] = None
    max_combo: Optional[int] = None


class ScanResult(BaseModel):
    label: str
    session: str
    symbols_scanned: int
    duration_s: float
    total: int = Field(description="Setups found (at most top_n)")
    setups: List[ScanSetup] = []


class ScanJob(BaseModel):
    scan_id: str
    status: Literal["queued", "running", "complete", "failed"]
    universe: str
    params: ScanParams
    progress: Optional[ScanProgress] = None
    result: Optional[ScanResult] = Field(default=None, description="Present when status is complete")
    error: Optional[str] = Field(default=None, description="Present when status is failed; safe to show. \"Cancelled.\" after DELETE /v1/scans/{scan_id}")
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# ---- scan history & historical research (P1-67, Pro) --------------------------------------------
class RunSummary(BaseModel):
    id: int
    name: Optional[str] = None
    label: Optional[str] = Field(default=None, description="SP500, NASDAQ, Combo, US Market, Watchlist (…), Search: X, 3-Step | …")
    row_count: Optional[int] = None
    duration_s: Optional[float] = None
    is_snapshot: bool = False
    created_at: Optional[str] = None


class RunDetail(RunSummary):
    total: int
    max_results: int
    limited: bool
    setups: List[ScanSetup] = []


class TrackRecordSummary(BaseModel):
    ranking: Literal["breakout", "prebreakout"]
    ranking_label: str
    horizon_days: int
    avg_excess_return: Optional[float] = Field(default=None, description="Mean return vs SPY (0.012 = +1.2%)")
    median_excess_return: Optional[float] = None
    win_rate: Optional[float] = Field(default=None, description="Share of picks that beat SPY")
    sample_size: Optional[int] = None
    runs_used: Optional[int] = None
    top_n: Optional[int] = None
    benchmark: str = "SPY"
    computed_at: Optional[str] = None
    sufficient: bool = Field(description="False while sample_size < min_sample_size (show 'still building')")


class TrackRecord(BaseModel):
    disclaimer: str
    min_sample_size: int
    summaries: List[TrackRecordSummary] = []


class TrackRecordDay(BaseModel):
    day: str
    avg_excess_return: Optional[float] = None


class EarningsItem(BaseModel):
    ticker: str
    earnings_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    days_until: Optional[int] = None
    time: Optional[str] = Field(default=None, description="bmo / amc / … when known")


class BriefIndex(BaseModel):
    label: str
    last: Optional[float] = None
    chg_pct: Optional[float] = None


class BriefMover(BaseModel):
    ticker: str
    chg_pct: Optional[float] = None


class BriefGapper(BaseModel):
    ticker: str
    last: Optional[float] = None
    chg_pct: Optional[float] = None
    gap_pct: Optional[float] = None
    earnings_days: Optional[int] = Field(default=None, description="Earnings in N days, when imminent")


class BriefPick(BaseModel):
    ticker: str
    prob: Optional[float] = None
    earnings_days: Optional[int] = None


class Brief(BaseModel):
    available: bool = Field(description="False until the day's first scan snapshot exists")
    snapshot_time: Optional[str] = None
    phase: Optional[str] = Field(default=None, description="premarket / regular / afterhours / closed")
    market: List[BriefIndex] = []
    breadth: Optional[Dict[str, int]] = Field(default=None, description="{advancers, decliners} in the snapshot")
    sectors: List[Dict[str, Any]] = Field(default=[], description="[{sector, chg_pct}] best first")
    opportunities: List[Dict[str, Any]] = Field(default=[], description="Top opportunities with movement since the "
                                                "previous snapshot (same objects as the web's Market Brief)")
    has_previous_snapshot: bool = False
    gappers: List[BriefGapper] = []
    gainers: List[BriefMover] = []
    losers: List[BriefMover] = []
    golden_crosses: List[str] = []
    top_breakout_scores: List[Dict[str, Any]] = Field(default=[], description="[{ticker, score}]")
    prebreakout_picks: List[BriefPick] = Field(default=[], description="Premium; empty below Premium")
    prebreakout_locked: bool = False
    earnings_today: List[str] = []


class DayTraderRow(BaseModel):
    ticker: str
    open: Optional[float] = None
    last: Optional[float] = None
    change_dollar: Optional[float] = None
    chg_pct: Optional[float] = None
    gap_pct: Optional[float] = None
    vwap: Optional[float] = None
    vs_vwap_pct: Optional[float] = None
    rvol: Optional[float] = None
    volume: Optional[float] = None
    adx: Optional[float] = None
    supertrend: Optional[float] = None
    supertrend_direction: Optional[Any] = None
    ewo: Optional[float] = None
    day_trade_score: float = Field(description="Intraday momentum score (move, VWAP alignment, gap, volume)")

    model_config = {"extra": "allow"}   # extra live fields (EMA cross, ranges, data source) pass through


class DayTrader(BaseModel):
    state: Literal["premarket", "open", "afterhours", "closed"]
    source: str
    symbols: List[str]
    missing: int = Field(description="Symbols with no live quote")
    as_of: str
    rows: List[DayTraderRow] = []


class StairSteppers(BaseModel):
    checked: List[str]
    matches: List[Dict[str, Any]] = Field(default=[], description="Symbols that pass the filters (r2, trend, pullback…)")
    all: List[Dict[str, Any]] = []


class AIText(BaseModel):
    run_id: Optional[int] = Field(default=None, description="The scan the text is about")
    ticker: Optional[str] = None
    snapshot_time: Optional[str] = None
    text: Optional[str] = Field(default=None, description="Markdown; null when there is nothing to explain")


class AIChatAnswer(BaseModel):
    run_id: Optional[int] = None
    answer: Optional[str] = None


class JournalTrade(BaseModel):
    id: int
    ticker: str
    entry_price: Optional[float] = None
    shares: Optional[int] = None
    source: Optional[str] = Field(default=None, description="scan, paper or api")
    entered_at: Optional[str] = None
    exit_price: Optional[float] = None
    closed_at: Optional[str] = None
    open: bool
    mark: Optional[float] = Field(default=None, description="Live price for open trades, exit price for closed")
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class Journal(BaseModel):
    trades: List[JournalTrade] = []
    stats: Optional[Dict[str, Any]] = Field(default=None, description="{closed, wins, avg_return_pct}; null before any closed trade")


class TradePlan(BaseModel):
    ticker: str
    entry: float
    stop: float
    stop_pct: float
    targets: List[float]
    target_r: List[float]
    risk_per_share: float
    shares: int
    risk_budget: float


class PaperStatus(BaseModel):
    connected: bool
    connected_at: Optional[str] = None
    account: Optional[Dict[str, Any]] = Field(default=None, description="{status, buying_power, cash} from Alpaca; keys are never returned")


class PaperActivity(BaseModel):
    connected: bool
    positions: List[Dict[str, Any]] = []
    positions_available: bool = True
    orders: List[Dict[str, Any]] = []


class PaperOrder(BaseModel):
    order_id: Optional[str] = None
    status: str
    ticker: str
    qty: int
    filled_avg_price: Optional[Any] = None
