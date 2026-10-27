"""Run 62 — front door & trust layer (UI/product only).

No live market data, database or Streamlit runtime needed: the pure helpers are
tested directly, and Streamlit-facing helpers use a tiny fake `st`.
"""
import datetime as dt
import unittest
from pathlib import Path
from types import SimpleNamespace

try:  # the Streamlit app was retired; its modules are gone
    from ui import chrome, landing, methodology, results_empty, safe_errors
except ImportError:
    chrome = landing = methodology = results_empty = safe_errors = None
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
                     pc.LAUNCH_DISCLOSURE, *[d for _t, d in pc.PILLARS],
                     *[d for _t, d in pc.TRUST_POINTS]):
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


class LandingTests(unittest.TestCase):


    def test_auth_uses_hero_instead_of_full_width_logo(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertIn("render_signed_out_hero()", src)
        self.assertIn("render_signed_out_details()", src)
        self.assertNotIn("st.columns([2, 1, 2])", src)
        self.assertIn('st.tabs(["Sign in", "Create account"])', src)


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


class PerformancePresentationTests(unittest.TestCase):


    def test_market_brief_scorecard_is_labelled_historical(self):
        src = (ROOT / "ui" / "market_brief.py").read_text()
        self.assertNotIn("### 📈 HSF Signal Performance — 7d", src)
        self.assertIn("flagged-signal outcomes (last 7 days)", src)
        self.assertIn('best_r is not None and bool(st.session_state.get("is_admin"))', src)

    def test_stock_intelligence_rates_are_labelled_historical(self):
        src = (ROOT / "ui" / "stock_intelligence.py").read_text()
        self.assertIn('f"{HISTORICAL_RESEARCH_LABEL} (descriptive)"', src)
        self.assertNotIn('st.markdown("#### Historical context")', src)


class ChromeTests(unittest.TestCase):
        # Phones use the ☰ Menu only (Run 80), so the "»" expand button is hidden
        # there — and only there (checked against desktop_css above).


    def test_tagline_free_logo_mark(self):
        self.assertTrue((ROOT / "assets" / "hsfinest_logo_mark.png").exists())
        self.assertTrue((ROOT / "assets" / "hsfinest_logo_512.png").exists())   # original kept


if __name__ == "__main__":
    unittest.main()


class LogoutMessageTests(unittest.TestCase):
    """Logging out reruns before the browser drops the cookie, so the next run
    finds a deleted session. It must say "signed out", not "expired"."""

    def test_delete_session_marks_signed_out(self):
        import importlib.util
        from unittest import mock

        if False:
            self.skipTest("streamlit not installed")
        from ui import auth_sessions

        fake_state = {}
        with mock.patch.object(auth_sessions.st, "session_state", fake_state), \
                mock.patch.object(auth_sessions, "get_neon_conn", None):
            auth_sessions.delete_session("abc")
        self.assertTrue(fake_state.get(auth_sessions.SIGNED_OUT_FLAG))

    def test_restore_branch_uses_the_same_flag(self):
        auth = (ROOT / "ui" / "auth.py").read_text()
        sessions = (ROOT / "ui" / "auth_sessions.py").read_text()
        self.assertIn('SIGNED_OUT_FLAG = "_hsf_signed_out"', sessions)
        self.assertIn('st.session_state.pop("_hsf_signed_out", False)', auth)
        self.assertIn("You've signed out.", auth)

    def test_new_sign_in_clears_the_flag(self):
        import importlib.util
        from unittest import mock

        if False:
            self.skipTest("streamlit not installed")
        from ui import auth_sessions

        fake_state = {auth_sessions.SIGNED_OUT_FLAG: True}
        with mock.patch.object(auth_sessions.st, "session_state", fake_state), \
                mock.patch.object(auth_sessions, "get_neon_conn", None):
            auth_sessions.create_session("someone")
        self.assertNotIn(auth_sessions.SIGNED_OUT_FLAG, fake_state)
