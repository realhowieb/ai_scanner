"""Run 62 — front door & trust layer (UI/product only).

No live market data, database or Streamlit runtime needed: the pure helpers are
tested directly, and Streamlit-facing helpers use a tiny fake `st`.
"""
import datetime as dt
import unittest
from pathlib import Path
from types import SimpleNamespace

from ui import chrome, landing, methodology, results_empty, safe_errors
from ui import product_copy as pc
from ui import trust_banner as tb

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc


def _health(generated_at, **statuses):
    subs = {name: {"status": "HEALTHY"} for name in
            ("universe", "scanner", "market_data", "database", "maturation", "research_capture",
             "cohort_parity", "forward_evidence", "workflows", "artifact_freshness")}
    for name, st in statuses.items():
        subs[name] = {"status": st}
    subs["universe"]["metrics"] = {"symbol_count": 11533}
    return {"generated_at": generated_at.isoformat(), "system_status": "HEALTHY", "subsystems": subs}


# Monday 2026-09-28 12:40 ET, during the session.
MON_MIDDAY = dt.datetime(2026, 9, 28, 16, 40, tzinfo=UTC)


class ProductCopyTests(unittest.TestCase):
    def test_positioning_is_factual_and_claim_free(self):
        for text in (pc.TAGLINE, pc.POSITIONING_ONE_LINE, pc.POSITIONING_SHORT, pc.POSITIONING_LONG,
                     pc.PAGE_TITLE, pc.EMAIL_SIGNOFF, pc.DISCLAIMER, pc.HISTORICAL_RESEARCH_NOTE,
                     *[d for _t, d in pc.PILLARS], *[d for _t, d in pc.TRUST_POINTS]):
            self.assertEqual(pc.find_prohibited_claims(text), [], text)
        self.assertIn("continuously scans the U.S. stock market", pc.POSITIONING_ONE_LINE)

    def test_guard_catches_promotional_claims(self):
        self.assertTrue(pc.find_prohibited_claims("Scan. Analyze. Trade. Win."))
        self.assertTrue(pc.find_prohibited_claims("Our proven edge beats the market"))
        self.assertTrue(pc.find_prohibited_claims("Guaranteed profitable signals"))
        self.assertTrue(pc.find_prohibited_claims("Which alerts pay off"))

    def test_guard_allows_disclaimers(self):
        self.assertEqual(pc.find_prohibited_claims("Past performance does not guarantee future results."), [])
        self.assertEqual(pc.find_prohibited_claims("Not a trading win rate."), [])

    def test_old_tagline_is_gone_from_the_app(self):
        for path in [ROOT / "app.py", *(ROOT / "ui").glob("*.py"), *(ROOT / "pages").glob("*.py")]:
            with self.subTest(path=path.name):
                self.assertNotIn("Trade. Win", path.read_text())


class LandingTests(unittest.TestCase):
    def test_hero_explains_hsf_before_the_form(self):
        h = landing.hero_html("")
        self.assertIn(pc.PRODUCT_NAME, h)
        self.assertIn(pc.TAGLINE, h)
        self.assertIn(pc.POSITIONING_SHORT, h)
        self.assertIn("Sign in or create a free account", h)
        self.assertEqual(pc.find_prohibited_claims(h), [])

    def test_hero_logo_is_small_on_phones(self):
        h = landing.hero_html("data:image/png;base64,AAAA")
        self.assertIn('alt="HSFinest.AI logo"', h)
        self.assertIn("@media (max-width:520px)", h)
        self.assertIn("width:64px", h)

    def test_details_sections_and_labelled_example(self):
        d = landing.details_html()
        for heading in ("What HSF does", "What a result looks like", "Why you can trust what you see"):
            self.assertIn(heading, d)
        for title, _desc in pc.PILLARS:
            self.assertIn(f"<h3>{title}</h3>", d)
        self.assertIn("not live data or recommendations", d)
        self.assertIn(pc.DISCLAIMER, d)
        self.assertEqual(pc.find_prohibited_claims(d), [])

    def test_example_rows_use_placeholder_names(self):
        for name, score, _setup, _why in landing.EXAMPLE_ROWS:
            self.assertTrue(name.startswith("Stock "))
            self.assertTrue(0 <= score <= 100)

    def test_landing_grids_do_not_force_many_columns(self):
        d = landing.details_html()
        self.assertIn("repeat(auto-fit,minmax(150px,1fr))", d)   # wraps to 2 columns at 375px

    def test_auth_uses_hero_instead_of_full_width_logo(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertIn("render_signed_out_hero()", src)
        self.assertIn("render_signed_out_details()", src)
        self.assertNotIn("st.columns([2, 1, 2])", src)
        self.assertIn('st.tabs(["Sign in", "Create account"])', src)


class MethodologyTests(unittest.TestCase):
    def test_sections_cover_the_required_topics(self):
        headings = [h for h, _b in methodology.METHODOLOGY_SECTIONS]
        for needed in ("Market coverage", "What HSF Score means", "Why a stock appears",
                       "Data freshness", "How HSF evaluates itself"):
            self.assertIn(needed, headings)

    def test_hsf_score_is_a_ranking_not_a_probability(self):
        body = dict(methodology.METHODOLOGY_SECTIONS)["What HSF Score means"]
        self.assertIn("opportunity-ranking score", body)
        self.assertIn("**not** a probability of profit", body)
        self.assertIn("**not** an expected return", body)

    def test_no_hardcoded_universe_size_or_claims(self):
        text = " ".join(b for _h, b in methodology.METHODOLOGY_SECTIONS)
        self.assertNotRegex(text, r"\b1\d,?\d{3}\b")
        self.assertEqual(pc.find_prohibited_claims(text), [])

    def test_methodology_page_is_public_and_linked(self):
        page = (ROOT / "pages" / "methodology.py").read_text()
        self.assertIn("render_methodology()", page)
        self.assertNotIn("st.stop()", page)            # readable signed out
        self.assertIn("pages/methodology.py", (ROOT / "ui" / "footer.py").read_text())
        self.assertIn("pages/methodology.py", (ROOT / "ui" / "landing.py").read_text())


class TrustBannerTests(unittest.TestCase):
    RUNS = [
        {"username": "cron", "label": "US_MARKET", "row_count": 100,
         "created_at": dt.datetime(2026, 9, 28, 16, 35, tzinfo=UTC)},
        {"username": "cron", "label": "US_MARKET", "row_count": 100,
         "created_at": dt.datetime(2026, 9, 28, 13, 35, tzinfo=UTC)},
        {"username": "alice", "label": "US_MARKET", "row_count": 7,
         "created_at": dt.datetime(2026, 9, 28, 16, 39, tzinfo=UTC)},   # a user's own scan
        {"username": "cron", "label": "postmarket", "row_count": 100,
         "created_at": dt.datetime(2026, 9, 28, 16, 38, tzinfo=UTC)},   # not the full market
    ]

    def test_real_data_rendering(self):
        info = tb.build_trust_info(self.RUNS, _health(MON_MIDDAY - dt.timedelta(minutes=30)), MON_MIDDAY)
        self.assertTrue(info["available"])
        self.assertEqual(info["last_scan_at"], dt.datetime(2026, 9, 28, 16, 35, tzinfo=UTC))
        parts = tb.banner_parts(info)
        self.assertIn("Last scan 12:35 PM ET (5 min ago)", parts)
        self.assertIn("Universe 11,533 tradable stocks", parts)
        self.assertIn("100 ranked setups", parts)
        self.assertIn("Market open", parts)
        html = tb.banner_html(info)
        self.assertIn("System: Operational", html)
        self.assertIn('role="status"', html)

    def test_fallback_when_no_scan_or_health(self):
        info = tb.build_trust_info([], None, MON_MIDDAY)
        self.assertFalse(info["available"])
        parts = tb.banner_parts(info)
        self.assertEqual(parts[0], tb.UNAVAILABLE_TEXT)
        html = tb.banner_html(info)
        self.assertNotIn("0 stocks", html)
        self.assertNotIn("Universe", html)
        self.assertIn("Status unavailable", html)

    def test_zero_or_missing_counts_are_not_shown(self):
        runs = [{"username": "cron", "label": "US_MARKET", "row_count": 0,
                 "created_at": MON_MIDDAY - dt.timedelta(hours=1)}]
        info = tb.build_trust_info(runs, None, MON_MIDDAY)
        self.assertIsNone(info["result_count"])
        self.assertNotIn("0 ranked setups", " ".join(tb.banner_parts(info)))

    def test_status_mapping_for_normal_users(self):
        fresh = MON_MIDDAY - dt.timedelta(hours=1)
        self.assertEqual(tb.user_status(_health(fresh), MON_MIDDAY)["label"], "Operational")
        self.assertEqual(tb.user_status(_health(fresh, scanner="DEGRADED"), MON_MIDDAY)["label"], "Limited")
        self.assertEqual(tb.user_status(_health(fresh, database="ACTION_REQUIRED"), MON_MIDDAY)["label"],
                         "Service issue")
        self.assertEqual(tb.user_status(None, MON_MIDDAY)["label"], "Status unavailable")

    def test_research_internals_never_affect_or_appear_in_status(self):
        fresh = MON_MIDDAY - dt.timedelta(hours=1)
        h = _health(fresh, maturation="DEGRADED", cohort_parity="ACTION_REQUIRED",
                    forward_evidence="ACTION_REQUIRED", research_capture="DEGRADED")
        self.assertEqual(tb.user_status(h, MON_MIDDAY)["level"], "ok")
        html = tb.banner_html(tb.build_trust_info(self.RUNS, h, MON_MIDDAY)).lower()
        for word in ("maturation", "forward", "cohort", "recovery", "incident", "epoch", "database"):
            self.assertNotIn(word, html)

    def test_stale_snapshot_is_status_unavailable(self):
        old = MON_MIDDAY - dt.timedelta(days=4)
        self.assertEqual(tb.user_status(_health(old), MON_MIDDAY)["level"], "unknown")

    def test_friday_night_snapshot_still_valid_over_the_weekend(self):
        fri_night = dt.datetime(2026, 9, 25, 23, 45, tzinfo=UTC)
        sunday = dt.datetime(2026, 9, 27, 18, 0, tzinfo=UTC)
        monday_open = dt.datetime(2026, 9, 28, 13, 0, tzinfo=UTC)
        self.assertEqual(tb.user_status(_health(fri_night), sunday)["level"], "ok")
        self.assertEqual(tb.user_status(_health(fri_night), monday_open)["level"], "ok")

    def test_session_labels(self):
        self.assertEqual(tb.market_session_label(dt.datetime(2026, 9, 27, 15, 0, tzinfo=UTC)), "Market closed")
        self.assertEqual(tb.market_session_label(dt.datetime(2026, 9, 28, 12, 0, tzinfo=UTC)), "Pre-market")
        self.assertEqual(tb.market_session_label(MON_MIDDAY), "Market open")
        self.assertEqual(tb.market_session_label(dt.datetime(2026, 9, 28, 21, 0, tzinfo=UTC)), "After hours")

    def test_older_scan_shows_date_and_age(self):
        fri = dt.datetime(2026, 9, 25, 19, 35, tzinfo=UTC)
        info = tb.build_trust_info([{"username": "cron", "label": "US_MARKET", "row_count": 100,
                                     "created_at": fri}], None, dt.datetime(2026, 9, 27, 15, 0, tzinfo=UTC))
        self.assertEqual(info["last_scan_text"], "Fri Sep 25, 3:35 PM ET")
        self.assertEqual(info["last_scan_age"], "43 h ago")

    def test_status_is_not_colour_alone(self):
        for level, (label, shape) in tb.STATUS_LABELS.items():
            self.assertTrue(label and shape, level)

    def test_banner_is_on_scanner_and_market_brief_only(self):
        self.assertIn("render_trust_banner()", (ROOT / "app.py").read_text())
        self.assertIn("render_trust_banner()", (ROOT / "pages" / "brief.py").read_text())
        self.assertNotIn("render_trust_banner", (ROOT / "pages" / "kalshi.py").read_text())


class _FakeSt:
    def __init__(self, admin=False):
        self.session_state = {"is_admin": admin}
        self.calls = []

    def _rec(self, name):
        return lambda *a, **k: self.calls.append((name, a))

    def __getattr__(self, name):
        if name in ("error", "warning", "info", "code", "caption"):
            return self._rec(name)
        raise AttributeError(name)

    def expander(self, label, expanded=False):
        self.calls.append(("expander", (label,)))
        return SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *a: False)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class SafeErrorTests(unittest.TestCase):
    def setUp(self):
        self._orig = safe_errors.st

    def tearDown(self):
        safe_errors.st = self._orig

    def _fake(self, admin):
        f = _FakeSt(admin)
        f.expander = lambda label, expanded=False: (f.calls.append(("expander", (label,))), _Ctx())[1]
        safe_errors.st = f
        return f

    def test_normal_user_sees_plain_message_only(self):
        f = self._fake(admin=False)
        with self.assertLogs("hsf.ui", level="ERROR") as logs:
            safe_errors.show_error("the market brief", KeyError("DATABASE_URL secret missing"))
        shown = " ".join(str(a) for _n, a in f.calls)
        self.assertIn("HSF couldn't load the market brief right now. Try again shortly.", shown)
        self.assertNotIn("KeyError", shown)
        self.assertNotIn("DATABASE_URL", shown)
        self.assertFalse(any(n in ("code", "expander") for n, _a in f.calls))
        self.assertTrue(any("market brief" in line for line in logs.output))   # logged, not swallowed

    def test_admin_sees_technical_details(self):
        f = self._fake(admin=True)
        with self.assertLogs("hsf.ui", level="ERROR"):
            safe_errors.show_error("the market brief", ValueError("boom"))
        self.assertIn(("expander", ("Technical details (admin)",)), f.calls)
        self.assertIn(("code", ("ValueError: boom",)), f.calls)

    def test_custom_message_and_level(self):
        f = self._fake(admin=False)
        with self.assertLogs("hsf.ui", level="ERROR"):
            safe_errors.show_error("x", RuntimeError("r"), level="error", message="Try later.")
        self.assertEqual(f.calls[0], ("error", ("Try later.",)))

    def test_startup_problem_text_detail_is_admin_only(self):
        f = self._fake(admin=False)
        with self.assertLogs("hsf.ui", level="ERROR"):
            safe_errors.show_startup_problem("Import error: ModuleNotFoundError: No module named 'x'")
        shown = " ".join(str(a) for _n, a in f.calls)
        self.assertIn(safe_errors.STARTUP_MESSAGE, shown)
        self.assertNotIn("ModuleNotFoundError", shown)

    def test_no_raw_exception_rendering_left_on_user_pages(self):
        for rel in ("app.py", "ui/results_tabs.py", "ui/alerts.py", "pages/brief.py", "pages/stock.py",
                    "pages/alerts.py", "pages/journal.py", "pages/kalshi.py", "pages/day_trader.py",
                    "pages/watchlists.py", "ui/personal_watchlist.py", "ui/paper_trade.py"):
            src = (ROOT / rel).read_text()
            with self.subTest(rel=rel):
                self.assertNotIn("st.exception(", src)
                self.assertNotRegex(src, r'caption\(f"\{type\((e|exc)\)\.__name__\}: \{(e|exc)\}"\)')


class EmptyStateTests(unittest.TestCase):
    def test_no_session_scan_vs_no_matches_are_distinct(self):
        self.assertEqual(results_empty.results_empty_message(None), results_empty.NO_SESSION_SCAN_MESSAGE)
        self.assertEqual(results_empty.results_empty_message([]), results_empty.NO_MATCHES_MESSAGE)
        self.assertNotEqual(results_empty.NO_SESSION_SCAN_MESSAGE, results_empty.NO_MATCHES_MESSAGE)
        # neither is confused with "no market scan available" (the trust banner's fallback)
        self.assertNotIn(tb.UNAVAILABLE_TEXT, results_empty.NO_SESSION_SCAN_MESSAGE)

    def test_tab_label_never_says_zero_rows(self):
        self.assertEqual(results_empty.results_tab_label(None), "📊 Your scan results")
        self.assertEqual(results_empty.results_tab_label([]), "📊 Your scan results (no matches)")
        self.assertEqual(results_empty.results_tab_label([1, 2]), "📊 Latest scan results (2 rows)")


class PerformancePresentationTests(unittest.TestCase):
    def test_no_cherry_picked_headline(self):
        src = (ROOT / "ui" / "track_record.py").read_text()
        summary = src[src.index("def _render_summary_metrics"):src.index("def _render_summary_table")]
        self.assertNotIn(".metric(", summary)                   # no headline tiles at all
        self.assertNotIn("_best_summary(", summary)             # the max is not used for display
        self.assertIn("HISTORICAL_RESEARCH_NOTE", src)
        self.assertIn("def _best_summary", src)                 # analytics preserved

    def test_backtest_badge_is_admin_only_on_results(self):
        src = (ROOT / "ui" / "results.py").read_text()
        self.assertIn("if is_admin_view:  # Run 62", src)
        self.assertIn("render_track_record_badge()", src)

    def test_market_brief_scorecard_is_labelled_historical(self):
        src = (ROOT / "ui" / "market_brief.py").read_text()
        self.assertNotIn("### 📈 HSF Signal Performance — 7d", src)
        self.assertIn("flagged-signal outcomes (last 7 days)", src)
        self.assertIn('best_r is not None and bool(st.session_state.get("is_admin"))', src)

    def test_stock_intelligence_rates_are_labelled_historical(self):
        src = (ROOT / "ui" / "stock_intelligence.py").read_text()
        self.assertIn('f"{HISTORICAL_RESEARCH_LABEL} (descriptive)"', src)
        self.assertNotIn('st.markdown("#### Historical context")', src)

    def test_research_tab_label(self):
        self.assertIn('"📚 Historical research"', (ROOT / "ui" / "results_tabs.py").read_text())

    def test_model_outputs_not_described_as_profit_probability(self):
        src = (ROOT / "ui" / "result_helpers.py").read_text()
        self.assertIn("Not a probability of profit", src)
        self.assertIn("not a validated probability of profit", src)


class ChromeTests(unittest.TestCase):
    def test_only_the_toolbar_actions_container_is_hidden(self):
        self.assertIn('[data-testid="stToolbarActions"]{display:none !important;}', chrome.CHROME_CSS)
        for keep in ("stSidebar", "stHeader", "stToolbar\"]", "stMainMenu", "stExpandSidebarButton"):
            self.assertNotIn(keep, chrome.CHROME_CSS)

    def test_applied_before_sign_in_and_on_sub_pages(self):
        self.assertIn("hide_developer_chrome()", (ROOT / "ui" / "app_boot.py").read_text())
        self.assertIn("hide_developer_chrome()", (ROOT / "ui" / "nav.py").read_text())

    def test_tagline_free_logo_mark(self):
        self.assertTrue((ROOT / "assets" / "hsfinest_logo_mark.png").exists())
        self.assertTrue((ROOT / "assets" / "hsfinest_logo_512.png").exists())   # original kept


if __name__ == "__main__":
    unittest.main()
