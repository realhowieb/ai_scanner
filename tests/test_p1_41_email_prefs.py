"""P1-41 — per-account email preferences and unsubscribe links.

Every scheduled email (morning digest, evening wrap, alert emails, intelligence
alerts, billing live alerts) respects the account's switch and carries an
unsubscribe link; the unsubscribe page never acts on load.
"""
import email
import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

HAS_ST = importlib.util.find_spec("streamlit") is not None
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


class Cursor:
    def __init__(self, db):
        self.db, self.rowcount, self.result = db, 0, None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        self.db.sql.append((s, params))
        low = s.lower()
        if low.startswith("select digest"):
            self.result = self.db.rows.get(params[0], {}).get("prefs")
        elif low.startswith("select unsub_token"):
            tok = self.db.rows.get(params[0], {}).get("token")
            self.result = (tok,) if params[0] in self.db.rows else None
        elif low.startswith("insert into hsf_email_prefs (user_id, unsub_token)"):
            row = self.db.rows.setdefault(params[0], {})
            row["token"] = row.get("token") or params[1]
        elif low.startswith("select user_id from hsf_email_prefs"):
            hit = [u for u, r in self.db.rows.items() if r.get("token") == params[0]]
            self.result = (hit[0],) if hit else None

    def fetchone(self):
        return self.result

    def close(self):
        pass


class Conn:
    def __init__(self, rows=None):
        self.rows, self.sql = dict(rows or {}), []

    def cursor(self):
        return Cursor(self)

    def commit(self):
        pass

    def close(self):
        pass


def with_db(db):
    return mock.patch("db.email_prefs.get_neon_conn", return_value=db)


class PrefsStoreTests(unittest.TestCase):
    def test_defaults_to_on_without_a_row_or_a_database(self):
        from db.email_prefs import get_prefs, wants_email

        with with_db(Conn()):
            self.assertEqual(get_prefs("a@example.com"), {"digest": True, "evening": True, "alerts": True})
        with mock.patch("db.email_prefs.get_neon_conn", return_value=None):
            self.assertTrue(wants_email("a@example.com", "digest"))

    def test_stored_switches_are_read(self):
        from db.email_prefs import get_prefs

        db = Conn({"a@example.com": {"prefs": (False, True, False)}})
        with with_db(db):
            self.assertEqual(get_prefs("A@Example.com"), {"digest": False, "evening": True, "alerts": False})

    def test_set_prefs_upserts_only_known_kinds(self):
        from db.email_prefs import set_prefs

        db = Conn()
        with with_db(db):
            self.assertTrue(set_prefs("a@example.com", digest=False, bogus=True))
            self.assertFalse(set_prefs("a@example.com", bogus=True))
        upsert = [s for s, _ in db.sql if s.startswith("INSERT INTO hsf_email_prefs (user_id, digest)")]
        self.assertEqual(len(upsert), 1)
        self.assertIn("ON CONFLICT (user_id) DO UPDATE SET digest = EXCLUDED.digest", upsert[0])

    def test_token_is_created_once_and_maps_back_to_the_account(self):
        from db.email_prefs import unsubscribe_token, unsubscribe_url, user_for_token

        db = Conn()
        with with_db(db):
            t1 = unsubscribe_token("a@example.com")
            t2 = unsubscribe_token("a@example.com")
            self.assertEqual(t1, t2)
            self.assertGreaterEqual(len(t1), 30)
            self.assertEqual(user_for_token(t1), "a@example.com")
            self.assertIsNone(user_for_token("nope"))
            self.assertIsNone(user_for_token("x" * 65))
            url = unsubscribe_url("a@example.com", "digest")
        self.assertTrue(url.endswith(f"/unsubscribe?t={t1}&k=digest"))


class EmailFooterTests(unittest.TestCase):
    def sent_message(self, fn, *args, **kw):
        import ui.email_utils as eu

        server = mock.MagicMock()
        server.__enter__.return_value = server
        with mock.patch("config.SMTP_HOST", "smtp.test"), mock.patch("config.SMTP_USER", "u"), \
             mock.patch("config.SMTP_PASS", "p"), mock.patch("smtplib.SMTP", return_value=server), \
             redirect_stdout(io.StringIO()):
            self.assertTrue(getattr(eu, fn)(*args, **kw))
        return email.message_from_string(server.sendmail.call_args.args[2])

    def test_digest_and_alert_carry_link_and_header(self):
        url = "https://hsf.example/unsubscribe?t=abc&k=digest"
        for fn, args in (("send_digest_email", ("a@example.com", "s", "<p>x</p>", "x")),
                         ("send_alert_email", ("a@example.com", "s", "body"))):
            msg = self.sent_message(fn, *args, unsubscribe_url=url)
            self.assertEqual(msg["List-Unsubscribe"], f"<{url}>", fn)
            bodies = " ".join(p.get_payload(decode=True).decode() for p in msg.walk() if not p.is_multipart())
            self.assertEqual(bodies.count(url), 2, fn)          # text + html footer

    def test_no_link_means_no_header(self):
        msg = self.sent_message("send_alert_email", "a@example.com", "s", "body")
        self.assertIsNone(msg["List-Unsubscribe"])


def _digest_run(opted_out=(), job="digest"):
    """Run the morning digest or evening wrap for three Pro accounts."""
    import scheduler.evening_wrap as ew
    import scheduler.morning_digest as md

    users = {"on@example.com": {"tier": "pro"}, "off@example.com": {"tier": "pro"},
             "admin@example.com": {"tier": "basic"}}
    sends = []

    def fake_send(to_address, **kw):
        sends.append((to_address, kw.get("unsubscribe_url")))
        return True

    common = [
        mock.patch("config.MORNING_DIGEST_ENABLED", True), mock.patch("config.MORNING_DIGEST_MAX_USERS", 500),
        mock.patch("db.users.load_users", return_value=users),
        mock.patch("db.users.is_admin_from_db", side_effect=lambda u: u == "admin@example.com"),
        mock.patch("db.email_verification.is_email_verified", return_value=True),
        mock.patch("db.watchlists.list_watchlists", return_value=[{"id": 1}]),
        mock.patch("db.watchlists.get_watchlist_tickers", return_value=["AAPL"]),
        mock.patch("market_data.build_day_trader_metrics", return_value=[]),
        mock.patch("ui.email_utils.send_digest_email", side_effect=fake_send),
        mock.patch("db.earnings.mark_earnings_refreshed_today"),
        mock.patch("db.email_prefs.wants_email", side_effect=lambda u, k: u not in opted_out),
        mock.patch("db.email_prefs.unsubscribe_url", side_effect=lambda u, k: f"https://x/unsubscribe?u={u}&k={k}"),
    ]
    if job == "digest":
        common += [mock.patch.object(md, n, return_value=v) for n, v in (
            ("_latest_snapshot_df", None), ("_market_gappers", []), ("_earnings_today", set()),
            ("_earnings_days_map", {}), ("_watchlist_notes", []), ("_compose", ("h", "t")))]
        run = lambda: md.run_morning_digest(force=True)  # noqa: E731
    else:
        common += [mock.patch.object(ew, n, return_value=v) for n, v in (
            ("_market_close_context", {}), ("_day_movers", ([], [])), ("_tomorrow_setups", ([], [])),
            ("_todays_events", []), ("_tomorrows_earnings", []), ("_compose_wrap", ("h", "t")))]
        common.append(mock.patch("scheduler.morning_digest._latest_snapshot_df", return_value=None))
        run = lambda: ew.run_evening_wrap(force=True)  # noqa: E731
    out = io.StringIO()
    with redirect_stdout(out):
        for p in common:
            p.start()
        try:
            run()
        finally:
            mock.patch.stopall()
    return sends, out.getvalue()


class ScheduledEmailTests(unittest.TestCase):
    def test_digest_skips_unsubscribed_and_links_the_rest(self):
        sends, log = _digest_run(opted_out=("off@example.com",))
        self.assertEqual(sorted(t for t, _ in sends), ["admin@example.com", "on@example.com"])
        self.assertTrue(all(u and u.endswith("&k=digest") for _, u in sends))
        self.assertIn("skipped: unsubscribed=1", log)

    def test_evening_wrap_matches_digest_rules(self):
        sends, log = _digest_run(opted_out=("off@example.com",), job="evening")
        self.assertEqual(sorted(t for t, _ in sends), ["admin@example.com", "on@example.com"])  # admin now eligible
        self.assertTrue(all(u and u.endswith("&k=evening") for _, u in sends))
        self.assertIn("[evening_wrap] sent 2 wrap(s); skipped: unsubscribed=1", log)

    def test_alert_runner_and_intelligence_alerts_respect_the_switch(self):
        import analytics.alert_evaluation as ae
        import scheduler.alert_runner as ar

        with mock.patch("db.email_prefs.wants_email", return_value=False):
            self.assertFalse(ar._alert_emails_on("a@example.com"))
            with mock.patch("ui.email_utils.send_alert_email") as send:
                self.assertFalse(ae._deliver_email("a@example.com", "copy"))
            send.assert_not_called()
        with mock.patch("db.email_prefs.wants_email", return_value=True), \
             mock.patch("db.email_prefs.unsubscribe_url", return_value="https://x/u"), \
             mock.patch("ui.email_utils.send_alert_email", return_value=True) as send:
            self.assertTrue(ae._deliver_email("a@example.com", "copy"))
        self.assertEqual(send.call_args.kwargs["unsubscribe_url"], "https://x/u")
        src = (ar.__file__ and open(ar.__file__).read())
        self.assertIn("and _alert_emails_on(user_id)", src)
        self.assertIn("unsubscribe_url=_alert_unsubscribe_link(user_id)", src)


class BillingLiveAlertTests(unittest.TestCase):
    def fake_conn(self, prefs_table):
        rows = [(1, "a@example.com", "aapl", 100.0, "above", "price", True, "pro", False, "tok")]
        state = {"sql": []}

        class C:
            def __init__(self):
                self.last = ""

            def execute(self, sql, params=()):
                self.last = sql
                state["sql"].append(" ".join(sql.split()))

            def fetchone(self):
                return (prefs_table,)

            def fetchall(self):
                return rows if prefs_table else [r[:8] for r in rows]

            def close(self):
                pass

        conn = mock.MagicMock()
        conn.cursor.side_effect = C
        return conn, state

    def test_reads_the_switch_when_the_table_exists(self):
        from billing_service import realtime_alerts as ra

        conn, state = self.fake_conn(True)
        with mock.patch.object(ra, "_apply_plan_limits", side_effect=lambda a: a):
            alerts = ra._due_price_alerts(conn)
        self.assertFalse(alerts[0]["alert_emails_on"])
        self.assertEqual(alerts[0]["unsub_token"], "tok")
        self.assertIn("LEFT JOIN hsf_email_prefs p", state["sql"][-1])

    def test_behaves_as_before_without_the_table(self):
        from billing_service import realtime_alerts as ra

        conn, state = self.fake_conn(False)
        with mock.patch.object(ra, "_apply_plan_limits", side_effect=lambda a: a):
            alerts = ra._due_price_alerts(conn)
        self.assertTrue(alerts[0]["alert_emails_on"])
        self.assertIsNone(alerts[0]["unsub_token"])
        self.assertNotIn("hsf_email_prefs", state["sql"][-1])

    def test_link_and_send_gate(self):
        from billing_service import realtime_alerts as ra

        self.assertIsNone(ra._unsubscribe_url(None))
        self.assertTrue(ra._unsubscribe_url("tok").endswith("/unsubscribe?t=tok&k=alerts"))
        src = open(ra.__file__).read()
        self.assertIn('and alert.get("alert_emails_on", True)', src)


PAGE = '''
import runpy, streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
runpy.run_path("pages/unsubscribe.py", run_name="__main__")
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class UnsubscribePageTests(unittest.TestCase):
    def render(self, query, prefs=None, user="a@example.com"):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(PAGE, default_timeout=60)
        for k, v in query.items():
            at.query_params[k] = v
        state = dict(prefs or {"digest": True, "evening": True, "alerts": True})
        saves = []

        def fake_set(u, **changes):
            saves.append((u, changes))
            state.update(changes)
            return True

        patches = [mock.patch("db.email_prefs.user_for_token", side_effect=lambda t: user if t == "good" else None),
                   mock.patch("db.email_prefs.get_prefs", side_effect=lambda u: dict(state)),
                   mock.patch("db.email_prefs.set_prefs", side_effect=fake_set)]
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        at.run()
        return at, saves

    def test_failed_save_says_so(self):
        at, _ = self.render({"t": "good", "k": "digest"})
        with mock.patch("db.email_prefs.set_prefs", return_value=False):
            at.button(key="unsub_one").click().run()
        self.assertIn("Couldn't save", at.error[0].value)

    def test_bad_token(self):
        at, saves = self.render({"t": "bad", "k": "digest"})
        self.assertFalse(at.exception)
        self.assertIn("isn't valid", at.error[0].value)
        self.assertEqual(saves, [])

    def test_never_acts_on_load_then_unsubscribes_on_click(self):
        at, saves = self.render({"t": "good", "k": "digest"})
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        self.assertEqual(saves, [])                                   # opening the link changes nothing
        self.assertIn("a***@example.com", " ".join(m.value for m in at.markdown))
        at.button(key="unsub_one").click().run()
        self.assertEqual(saves, [("a@example.com", {"digest": False})])
        self.assertIn("You're unsubscribed from the morning market digest", at.success[0].value)
        self.assertNotIn("unsub_one", [b.key for b in at.button])      # button gone after saving

    def test_unsubscribe_from_everything(self):
        at, saves = self.render({"t": "good", "k": "alerts"})
        at.button(key="unsub_all").click().run()
        self.assertEqual(saves, [("a@example.com", {"digest": False, "evening": False, "alerts": False})])
        self.assertIn("unsubscribed from all HSF emails", at.success[0].value)
        self.assertEqual(len(at.button), 0)


class SettingsSourceTests(unittest.TestCase):
    def test_settings_page_has_the_three_switches(self):
        src = open("pages/settings.py").read()
        self.assertIn("from db.email_prefs import KINDS, LABELS, get_prefs, set_prefs", src)
        self.assertIn('key=f"email_pref_{_kind}"', src)

    def test_unsubscribe_page_follows_page_conventions_and_stays_out_of_nav(self):
        from ui.nav import _NAV

        src = open("pages/unsubscribe.py").read()
        self.assertLess(src.index("st.set_page_config("), src.index("hide_developer_chrome()"))
        self.assertIn("render_page_logo", src)
        self.assertIn('render_page_header("Email preferences"', src)
        self.assertNotIn("pages/unsubscribe.py", [p for p, _l, _i in _NAV])


if __name__ == "__main__":
    unittest.main()
