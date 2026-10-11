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


class DataFreshness(BaseModel):
    state: Literal["fresh", "stale", "partial", "unavailable"]
    scan_state: Literal["fresh", "stale", "partial", "unavailable"]
    market_data_state: Literal["fresh", "stale", "partial", "unavailable"]
    last_successful_scan_at: Optional[str] = None
    scan_completed_at: Optional[str] = None
    market_data_at: Optional[str] = None
    expected_scan_at: Optional[str] = None
    calendar_covered: bool
    checked_at: str
    timestamp_basis: Literal["saved_scan"]
    market_data_note: Optional[str] = None


class Market(BaseModel):
    freshness: Optional[DataFreshness] = Field(default=None, description='Scan and underlying-data freshness; missing source times stay null')
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
    setups: List[Setup] = Field(default=[], description="Strong setups, HSF `threshold`+")
    ranked_floor: Optional[int] = Field(default=None, description="HSF floor of `also_ranked` (the ranked list, 40)")
    also_ranked: List[Setup] = Field(default=[], description="Next ranked names filling the card to 5 when fewer are strong")


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


class SystemStatus(BaseModel):
    level: Literal["ok", "limited", "issue", "unknown"]
    label: str = Field(description="Operational, Limited, Service issue or Status unavailable")


class IndexQuote(BaseModel):
    symbol: str = Field(description="SPY or QQQ")
    label: str
    last: float
    chg_pct: Optional[float] = Field(default=None, description="Percent vs the previous close")


class ScanLeader(BaseModel):
    ticker: str
    chg_pct: Optional[float] = Field(default=None, description="Percent vs the previous close, from the scan")
    last: Optional[float] = None
    volume: Optional[float] = Field(default=None, description="Shares traded in the session, from the scan")


class TapeQuote(BaseModel):
    symbol: str
    last: float
    chg_pct: Optional[float] = Field(default=None, description="Percent vs the previous close")


class Tape(BaseModel):
    """The scrolling price strip: market ETFs and a few large caps."""
    quotes: List[TapeQuote] = Field(default=[], description="In display order; empty when no quote is available")


class Snapshot(BaseModel):
    """The status strip and market snapshot (the Streamlit app's trust banner and
    "Today's Market Snapshot"). Each field is null when its source isn't available."""
    universe_symbols: Optional[int] = Field(default=None, description="Tradable U.S. stocks in the scan universe")
    ranked_count: Optional[int] = Field(default=None, description="Results in the latest full-market scan")
    status: SystemStatus
    indices: List[IndexQuote] = Field(default=[], description="SPY and QQQ; empty when no quote is available")
    top_gainer: Optional[ScanLeader] = Field(default=None, description="Biggest gain in the latest full-market scan")
    most_active: Optional[ScanLeader] = Field(default=None, description="Most shares traded in the latest full-market scan")


class ScoredTicker(BaseModel):
    ticker: str
    score: Optional[int] = Field(default=None, description="HSF Score in the latest market scan; null when it doesn't qualify")


class NewSince(BaseModel):
    marker: Optional[str] = Field(default=None, description="Store this and send it back as seen:baseline on the next visit")
    baseline_scan_at: Optional[str] = Field(default=None, description="The scan the latest one was compared with; null on a first visit")
    tickers: List[ScoredTicker] = Field(default=[], description="Names new in the latest market scan, strongest first (at most 50)")
    total: int = 0


class WatchlistCounts(BaseModel):
    tracked: int
    needs_attention: int
    strengthening: int
    fading: int


class WatchlistToday(BaseModel):
    watchlist_id: Optional[int] = Field(default=None, description="The default watchlist; null when the user has none")
    name: Optional[str] = None
    summary: Optional[WatchlistCounts] = Field(default=None, description="Across all the user's watchlists, as in the Streamlit app")
    in_scan: List[ScoredTicker] = Field(default=[], description="Watched names in the latest market scan, strongest first")
    missing: List[str] = Field(default=[], description="Watched names not in the latest market scan")


class TodayPersonal(BaseModel):
    new_since: Optional[NewSince] = None
    watchlist: Optional[WatchlistToday] = None
    errors: List[SectionError] = []


class Today(BaseModel):
    as_of: str
    market: Market
    before_open: Optional[SessionCard] = None
    top_setups: Optional[TopSetups] = None
    after_close: Optional[SessionCard] = None
    recap: Optional[Recap] = None
    snapshot: Optional[Snapshot] = None
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
    freshness: Optional[DataFreshness] = None
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


class WatchlistScanState(ScanSetup):
    """The ticker's row in the latest market scan's ranked setups."""
    rank: int = Field(description="Position in that scan's ranked setups (1 = top)")


class WatchlistItemDetail(WatchlistItem):
    latest: Optional[WatchlistScanState] = Field(default=None, description=(
        "GET /v1/watchlists/{id} only: the ticker's row in the latest market scan; null when it "
        "isn't a ranked setup there (or the scan couldn't be read)"))


class WatchlistDetail(Watchlist):
    items: List[WatchlistItemDetail] = []
    scan_at: Optional[str] = Field(default=None, description="GET only: the market scan behind items[].latest")
    scan_total: Optional[int] = Field(default=None, description="GET only: ranked setups in that scan")
    stale: Optional[bool] = Field(default=None, description="GET only: a scheduled scan was missed (see Market.stale)")


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
    id: int = Field(description="Row id within its source; use event_id as a unique key")
    event_id: str = Field(description="Unique across sources: 'alert:<id>' or 'rule:<id>'")
    source: Literal["alert", "rule"] = Field(description="alert: a ticker alert (/v1/alerts); rule: an alert rule")
    alert_id: Optional[int] = None
    ticker: Optional[str] = None
    message: str
    fired_at: Optional[str] = Field(default=None, description="When it triggered (same as triggered_at)")
    triggered_at: Optional[str] = None
    rule_id: Optional[int] = None
    watchlist_id: Optional[int] = None
    rule_type: Optional[str] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    trigger_value: Optional[float] = Field(default=None, description="The value that met the condition")
    previous_value: Optional[float] = Field(default=None, description="The rule's last seen value (transitions)")
    hsf_score: Optional[float] = None
    setup: Optional[str] = None
    market_data_as_of: Optional[str] = Field(default=None, description="Time of the market scan it was evaluated on")
    delivery: Dict[str, str] = Field(default={}, description=(
        "Per channel: delivered (in_app), sent, pending, failed (retried), skipped"))
    cursor: str = Field(description="Pass as ?cursor= to get the events after this one")


# ---- watchlist intelligence and alert rules ---------------------------------------------------------
class WatchlistIntelItem(BaseModel):
    ticker: str
    added_at: Optional[str] = None
    note: Optional[str] = None
    company_name: Optional[str] = Field(default=None, description="Not available from the scans yet (always null)")
    price: Optional[float] = Field(default=None, description="Last price in the latest market scan")
    price_change: Optional[float] = Field(default=None, description="Not available from the scans yet (always null)")
    price_change_pct: Optional[float] = None
    hsf_score: Optional[float] = Field(default=None, description="Null when the ticker isn't a ranked HSF setup")
    previous_hsf_score: Optional[float] = Field(default=None, description="In the previous market scan")
    score_change: Optional[float] = None
    rank: Optional[int] = Field(default=None, description="Position in the latest scan's ranked setups (1 = top)")
    previous_rank: Optional[int] = None
    rank_change: Optional[int] = Field(default=None, description="Places moved up since the previous scan (negative = down)")
    status: Optional[str] = None
    setup: Optional[str] = None
    signals: List[str] = []
    fading: Optional[bool] = None
    ranked: bool = Field(default=False, description="A ranked HSF setup in the latest scan")
    prebreakout: Optional[bool] = Field(default=None, description="Premium: the PreBreakout signal is on; null below Premium")
    prebreakout_score: Optional[float] = Field(default=None, description="Premium: PreBreakout probability %")
    prebreakout_rank_pct: Optional[float] = Field(default=None, description="Premium: 'top N%' of the model's scores")
    breakout_score: Optional[float] = None
    rvol: Optional[float] = Field(default=None, description="Volume vs 20-day average, from the scan")
    rsi: Optional[float] = Field(default=None, description="Not computed by the scans yet (always null)")
    ema_cross: Optional[Literal["golden", "death"]] = Field(default=None, description="EMA 9/21 cross state in the scan")
    in_latest_scan: bool
    freshness: Literal["fresh", "stale", "missing", "unavailable"] = Field(description=(
        "fresh: in the latest scan, which is on schedule; stale: a scheduled scan was missed; "
        "missing: not in the latest scan; unavailable: scan data couldn't be read"))
    active_alert_count: Optional[int] = Field(default=None, description=(
        "Enabled alert rules on this ticker or this watchlist, plus enabled ticker alerts"))


class Coverage(BaseModel):
    symbols: int
    enriched: int
    missing: int


class WatchlistIntelligence(BaseModel):
    watchlist_id: int
    name: str
    market_session: Literal["premarket", "open", "afterhours", "closed"]
    scan_available: bool
    last_scan_at: Optional[str] = None
    previous_scan_at: Optional[str] = None
    market_data_as_of: Optional[str] = Field(default=None, description="Prices and scores are as of this scan")
    stale: Optional[bool] = None
    scan_total: Optional[int] = None
    prebreakout_locked: bool
    coverage: Coverage
    unavailable_fields: List[str] = Field(description="Fields always null because no canonical source exists yet")
    items: List[WatchlistIntelItem] = []


class WatchlistChange(BaseModel):
    ticker: str
    event_type: str = Field(description="NEW_OPPORTUNITY, DROPPED, RISING, FALLING, STATUS_UPGRADE, STATUS_DOWNGRADE, "
                                        "FADING, SIGNAL_ADDED, SIGNAL_REMOVED")
    severity: Optional[str] = None
    previous_score: Optional[float] = None
    current_score: Optional[float] = None
    score_delta: Optional[float] = None
    previous_status: Optional[str] = None
    current_status: Optional[str] = None
    setup: Optional[str] = None
    signal: Optional[str] = Field(default=None, description="SIGNAL_ADDED/REMOVED: which signal (e.g. prebreakout)")
    previous_rank: Optional[int] = None
    rank: Optional[int] = None
    rank_change: Optional[int] = None


class WatchlistHeadline(BaseModel):
    ticker: str
    event_type: str


class WatchlistChanges(BaseModel):
    watchlist_id: int
    name: str
    last_scan_at: Optional[str] = None
    previous_scan_at: Optional[str] = None
    has_baseline: bool = Field(description="False until two market scans exist")
    changes: List[WatchlistChange] = []
    headline: List[WatchlistHeadline] = Field(default=[], description="The strongest change per ticker")
    alerts: List[AlertEvent] = Field(default=[], description="Your rule alerts on these tickers since the previous scan")


class AlertRule(BaseModel):
    id: int
    watchlist_id: Optional[int] = None
    ticker: Optional[str] = None
    rule_type: str
    operator: str
    threshold: Optional[float] = None
    value: Optional[str] = None
    enabled: bool
    delivery_channels: List[str]
    cooldown_seconds: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_evaluated_at: Optional[str] = None
    last_triggered_at: Optional[str] = None


class Capabilities(BaseModel):
    tier: str
    max_watchlists: int
    max_symbols_per_watchlist: Optional[int] = Field(default=None, description="Null: no plan limit is defined")
    max_symbols_per_request: int
    max_active_alerts: int = Field(description="Enabled alert rules plus enabled ticker alerts")
    alert_rule_types: List[str]
    delivery_channels: List[str]


class AlertRules(BaseModel):
    limit: int = Field(description="Active alerts this plan may have; rules and ticker alerts share it")
    used: int = Field(description="Enabled rules plus enabled ticker alerts")
    capabilities: Capabilities
    rules: List[AlertRule] = []


class RuleThreshold(BaseModel):
    min: float
    max: float
    default: Optional[float] = None


class AlertRuleType(BaseModel):
    type: str
    label: str
    description: str
    operator: str
    kind: Literal["level", "transition"]
    threshold: Optional[RuleThreshold] = None
    takes_value: bool = Field(description="SETUP_APPEARED: optional setup name, e.g. Breakout")
    default_cooldown_seconds: int
    available: bool = Field(description="False when the plan doesn't include it")


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
    provider: Literal["apns", "fcm", "expo", "webpush"]
    platform: Literal["ios", "android", "web"]
    device_name: Optional[str] = None
    app_version: Optional[str] = None
    created_at: Optional[str] = None
    last_seen_at: Optional[str] = None


class WebPushConfig(BaseModel):
    enabled: bool = Field(description="False until the server has VAPID keys")
    public_key: Optional[str] = Field(default=None, description="VAPID application server key (base64url)")


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
    freshness: Optional[DataFreshness] = None
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
    previous_close: Optional[float] = None
    close_today: Optional[float] = Field(default=None, description="Regular-session close (today's bar)")
    session_chg_pct: Optional[float] = Field(default=None, description="After hours / closed: regular-session "
                                             "change (close vs previous close)")
    ext_chg_pct: Optional[float] = Field(default=None, description="After hours / closed: last trade vs the close")
    day_trade_score: Optional[float] = Field(default=None, description="DT score 0-100: setup strength and signal "
                                             "agreement (null with too little evidence). Off hours it scores the "
                                             "completed session.")
    dt_quality: Literal["strong", "developing", "weak", "insufficient"] = "insufficient"
    dt_direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    dt_reasons: List[str] = Field(default=[], description="Signals behind the score")
    dt_conflicts: List[str] = Field(default=[], description="Signals that lower the score")
    quote_flags: List[str] = Field(default=[], description="Why the quote may be wrong: Extreme move, Far from "
                                   "VWAP, Stale quote. Flagged rows rank last.")

    model_config = {"extra": "allow"}   # extra live fields (EMA cross, ranges, data source) pass through


class DayTrader(BaseModel):
    state: Literal["premarket", "open", "afterhours", "closed"]
    source: str
    symbols: List[str]
    missing: int = Field(description="Symbols with no live quote")
    as_of: str
    rows: List[DayTraderRow] = []


class DayTraderSparklines(BaseModel):
    checked: List[str]
    series: Dict[str, List[float]] = Field(default={}, description="Latest-session 1-minute closes, oldest first")


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


class PlanTier(BaseModel):
    id: Literal["basic", "pro", "premium"]
    name: str
    price: str = Field(description='Display price, e.g. "$25/mo" or "Free"')
    yearly_price: Optional[str] = None
    tagline: str
    alert_limit: int
    highlights: List[str] = Field(description="What this plan adds over the one below it")


class PlanRow(BaseModel):
    label: str
    basic: Any = Field(description="true/false, or the alert limit for the alerts row")
    pro: Any
    premium: Any


class Plans(BaseModel):
    tiers: List[PlanTier]
    rows: List[PlanRow] = Field(description="The comparison table, derived from the real entitlement map")


class UnsubscribeState(BaseModel):
    email: str = Field(description="Masked, e.g. sa***@gmail.com")
    prefs: EmailPrefs


# --- Outcome Intelligence (/v1/outcomes/*) -----------------------------------------
# Returns are fractions (0.012 = +1.2%). Every aggregate carries its own counts; rates
# use the count named beside them. Nothing is zero-filled: missing data is null.

class OutcomeFilters(BaseModel):
    """The filters that produced this answer. All null/false = the complete dataset."""
    ticker: Optional[str] = None
    setup: Optional[str] = None
    signal: Optional[str] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    score_bucket: Optional[str] = None
    score_version: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    certified_only: bool = False
    matured_only: bool = False


class OutcomeInterval(BaseModel):
    low: float
    high: float


class OutcomeDateRange(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class OutcomeDataset(BaseModel):
    rows_loaded: int
    truncated: bool = Field(description="True when the row cap was hit (then the answer is not the full history)")
    loaded_at: str
    stale: bool = Field(description="True when the database could not be reached and the last good dataset was used")
    source: str


class OutcomeMetrics(BaseModel):
    horizon: int = Field(description="Trading days (1, 3 or 5)")
    sample_size: int = Field(description="Records in the group, any maturity")
    matured_count: int
    pending_count: int
    unavailable_count: int = Field(description="Outcome computed but no price was available; excluded")
    invalid_count: int = Field(description="Outcome timestamp at/before the observation; excluded")
    distinct_days: int = Field(description="Distinct entry days among matured records (observations on one day are correlated)")
    evidence_quality: Literal["INSUFFICIENT", "LIMITED", "MODERATE", "STRONG"]
    average_return: Optional[float] = None
    median_return: Optional[float] = None
    average_return_ci95: Optional[OutcomeInterval] = Field(default=None, description="Normal approximation; from 30 matured; assumes independence")
    win_count: int
    win_rate: Optional[float] = Field(default=None, description="Share of matured with return > 0")
    win_rate_ci95: Optional[OutcomeInterval] = Field(default=None, description="Wilson interval; assumes independence")
    benchmark_count: int = Field(description="Matured records with a benchmark return")
    average_benchmark_return: Optional[float] = None
    median_benchmark_return: Optional[float] = None
    average_excess_return: Optional[float] = None
    median_excess_return: Optional[float] = None
    benchmark_beat_count: int
    benchmark_beat_rate: Optional[float] = Field(default=None, description="Share of benchmark_count with excess > 0")
    benchmark_beat_rate_ci95: Optional[OutcomeInterval] = None
    mfe_count: int
    average_mfe: Optional[float] = None
    median_mfe: Optional[float] = None
    mae_count: int
    average_mae: Optional[float] = None
    median_mae: Optional[float] = None


class OutcomeCoverage(BaseModel):
    matured: int
    missing_benchmark: int
    benchmark_coverage: Optional[float] = None
    mfe_mae_available_for_horizon: bool
    missing_mfe_mae: int
    mfe_mae_coverage: Optional[float] = None


class _OutcomeEnvelope(BaseModel):
    filters: OutcomeFilters
    unit: Literal["signal_day", "observation"] = Field(
        description="signal_day = one record per ticker per entry day (the day's first observation); observation = every frozen row")
    horizon: Optional[int] = None
    raw_observations: int = Field(description="Frozen rows matching the filters before collapsing to the unit")
    date_range: OutcomeDateRange
    score_versions: Dict[str, int]
    warnings: List[str] = []
    evidence_thresholds: Dict[str, int] = Field(description="Minimum matured count per evidence label")
    disclaimer: str
    dataset: OutcomeDataset
    generated_at: str


class OutcomeSummary(_OutcomeEnvelope):
    total_observations: int
    matured_observations: int
    pending_observations: int
    unavailable_observations: int
    certified_observations: int
    metrics: OutcomeMetrics
    coverage: OutcomeCoverage


class OutcomeBucket(OutcomeMetrics):
    bucket: str
    min_score: int
    max_score: int


class OutcomeInversion(BaseModel):
    lower_bucket: str
    higher_bucket: str
    lower_value: float
    higher_value: float


class OutcomeMonotonicity(BaseModel):
    buckets_compared: List[str]
    monotonic: Optional[bool] = Field(default=None, description="null when fewer than two buckets have enough evidence")
    inversions: List[OutcomeInversion] = []


class OutcomeCalibration(BaseModel):
    min_matured_per_bucket: int
    metrics: Dict[str, OutcomeMonotonicity]


class OutcomeScores(_OutcomeEnvelope):
    buckets: List[OutcomeBucket]
    unbucketed_count: int
    calibration: OutcomeCalibration


class OutcomeHorizonRow(OutcomeMetrics):
    coverage: OutcomeCoverage


class OutcomeHorizons(_OutcomeEnvelope):
    horizons: List[OutcomeHorizonRow]


class OutcomeScoreDistribution(BaseModel):
    scored: int
    unscored: int
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    bucket_counts: Dict[str, int]


class OutcomeGroup(OutcomeMetrics):
    name: str
    score_distribution: OutcomeScoreDistribution
    supported_horizons: List[int]


class OutcomeGroups(_OutcomeEnvelope):
    group_by: Literal["setup", "signal"]
    groups_overlap: bool = Field(description="True for signals: one record can carry several")
    groups: List[OutcomeGroup]


class OutcomePoint(BaseModel):
    period_start: str
    observation_count: int
    matured_count: int
    pending_count: int
    evidence_quality: str
    average_return: Optional[float] = None
    median_return: Optional[float] = None
    average_excess_return: Optional[float] = None
    median_excess_return: Optional[float] = None
    win_rate: Optional[float] = None
    benchmark_beat_rate: Optional[float] = None
    benchmark_count: int


class OutcomeTimeseries(_OutcomeEnvelope):
    period: Literal["day", "week", "month"]
    points: List[OutcomePoint]


class OutcomeHorizonResult(BaseModel):
    horizon: int
    status: Literal["pending", "matured", "unavailable", "invalid"]
    raw_return: Optional[float] = None
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None


class OutcomeObservation(BaseModel):
    observation_id: Optional[str] = None
    ticker: str
    observed_at: Optional[str] = None
    entry_day: Optional[str] = None
    hsf_score: Optional[float] = Field(default=None, description="As frozen at signal time")
    score_bucket: Optional[str] = None
    score_version: Optional[str] = None
    setup: Optional[str] = None
    signals: List[str] = []
    status: Optional[str] = None
    prebreakout_prob: Optional[float] = None
    certified: bool
    benchmark_symbol: Optional[str] = None
    entry_price: Optional[float] = Field(default=None, description="Not stored by the outcome engine yet (always null)")
    outcome_price: Optional[float] = Field(default=None, description="Not stored by the outcome engine yet (always null)")
    observations_that_day: Optional[int] = Field(default=None, description="Frozen rows for this ticker on this entry day (they share one outcome)")
    outcomes: List[OutcomeHorizonResult]


class OutcomePage(BaseModel):
    page: int
    page_size: int
    total: int
    items: List[OutcomeObservation]


class OutcomeQuery(_OutcomeEnvelope):
    metrics: OutcomeMetrics
    coverage: OutcomeCoverage
    observations: OutcomePage


class OutcomeSymbol(_OutcomeEnvelope):
    ticker: str
    horizons: List[OutcomeMetrics]
    observations: OutcomePage
