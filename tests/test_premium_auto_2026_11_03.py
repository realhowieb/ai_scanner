"""Premium automations (2026-11-03): saved AI answers (db.premium_auto), the automation
jobs (services.premium_auto), the AI read in scheduled-scan emails and automatic stop and
target alerts for logged trades (api.trade_alerts)."""
import datetime as dt
import json
import sqlite3
import unittest
from unittest import mock

UTC = dt.timezone.utc


class _Store(unittest.TestCase):
    def setUp(self):
        from db import premium_auto

        self.conn = sqlite3.connect(":memory:")
        premium_auto.set_connection_factory(lambda: self.conn)
        self.addCleanup(premium_auto.set_connection_factory, None)
        self.store = premium_auto


class PlanTests(unittest.TestCase):
    def test_flag_card_and_upgrade(self):
        from domain.plans import FEATURE_MIN_TIER
        from domain.upgrades import upgrade_message
        from ui.pricing import plan_highlights

        self.assertEqual(FEATURE_MIN_TIER["can_auto_research"], "premium")
        prem = " | ".join(plan_highlights()["premium"])
        for want in ("ready when you look", "Sunday AI watchlist rundown", "Stop and target price alerts"):
            self.assertIn(want, prem)
        self.assertNotIn("Sunday AI watchlist rundown", " | ".join(plan_highlights()["pro"]))
        self.assertIn("Premium", upgrade_message("can_auto_research"))

    def test_account_deletion_covers_the_table(self):
        from db import account_deletion

        self.assertIn(("hsf_premium_auto", "username"), account_deletion.ACCOUNT_TABLES)


class StoreTests(_Store):
    def test_save_get_replace_list_delete_prune(self):
        s = self.store
        s.save("p@x.com", "watchlist_note", "3", json.dumps({"text": "one"}))
        s.save("p@x.com", "watchlist_note", "3", json.dumps({"text": "two"}))
        self.assertEqual(json.loads(s.get("p@x.com", "watchlist_note", "3")["body"])["text"], "two")
        self.assertIsNone(s.get("q@x.com", "watchlist_note", "3"))
        s.save("p@x.com", "watchlist_note", "4", "{}")
        self.assertEqual(len(s.list_kind("p@x.com", "watchlist_note")), 2)
        s.delete("p@x.com", "watchlist_note", "4")
        self.assertEqual(len(s.list_kind("p@x.com", "watchlist_note")), 1)
        s.save("*", "ticker_note", "9:AAA", "{}")
        self.assertEqual(s.prune("ticker_note", dt.datetime.now(UTC) + dt.timedelta(minutes=1)), 1)
        self.assertEqual(s.delete_all_for_user("p@x.com"), 1)

    def test_api_stored_helpers(self):
        from api import ai

        ai.store("p@x.com", "journal_coach", "latest", "- note", closed=6)
        got = ai.stored("p@x.com", "journal_coach", "latest")
        self.assertEqual((got["text"], got["closed"]), ("- note", 6))
        self.assertTrue(got["written_at"])
        ai.store("p@x.com", "journal_coach", "latest", None)          # nothing to save keeps the old one
        self.assertEqual(ai.stored("p@x.com", "journal_coach", "latest")["text"], "- note")
        self.assertIsNone(ai.stored("p@x.com", "journal_coach", "missing"))


class SavedAnswerTests(_Store):
    def test_brief_narrative_saved_and_stored_only(self):
        from api import ai

        core = {"data": {"snapshot_time": "2026-11-03T14:30:00Z"}}
        with mock.patch("api.market._brief_core", return_value=core), \
                mock.patch("ui.market_brief._brief_narrative_facts", return_value="facts"), \
                mock.patch("api.ai._ask", return_value="Markets were mixed.") as ask:
            self.assertEqual(ai.brief_narrative(None, stored_only=True)["text"], None)
            ask.assert_not_called()
            out = ai.brief_narrative("p@x.com")
            self.assertEqual(out["text"], "Markets were mixed.")
            again = ai.brief_narrative(None, stored_only=True)
        self.assertEqual((again["text"], again["auto"]), ("Markets were mixed.", True))
        self.assertEqual(ask.call_count, 1)

    def test_ticker_note_saved_for_the_latest_scan_only(self):
        from api import ai

        ai.store(ai.SHARED_USER, "ticker_note", "7:AAA", "AAA note")
        with mock.patch("api.today.market_runs", return_value=[{"id": 7}]):
            self.assertEqual(ai.ticker_note(None, "AAA", stored_only=True)["text"], "AAA note")
            self.assertIsNone(ai.ticker_note(None, "BBB", stored_only=True)["text"])
        with mock.patch("api.today.market_runs", return_value=[{"id": 8}]):
            self.assertIsNone(ai.ticker_note(None, "AAA", stored_only=True)["text"])   # a newer scan

    def test_latest_watchlist_note(self):
        from api import ai

        self.assertIsNone(ai.latest_watchlist_note("p@x.com", 3)["text"])
        ai.store("p@x.com", "watchlist_note", 3, "AAA rose.", snapshot_time="2026-11-03T14:00:00Z")
        out = ai.latest_watchlist_note("p@x.com", 3)
        self.assertEqual((out["text"], out["snapshot_time"], out["auto"]), ("AAA rose.", "2026-11-03T14:00:00Z", True))


class JobTimingTests(unittest.TestCase):
    def test_empty_brief_releases_claim_for_retry(self):
        from services import premium_auto as pa

        with mock.patch.object(pa, "brief_due", return_value=True), \
                mock.patch.object(pa, "_ai_ready", return_value=True), \
                mock.patch("api.market._brief_core", return_value={"data": {"snapshot_time": "scan-1"}}), \
                mock.patch("api.ai.stored", return_value=None), \
                mock.patch("api.ai.brief_narrative", return_value={"text": None}):
            release = mock.Mock()
            out = pa.run_brief(dt.datetime.now(UTC), claim=lambda *a: True, release=release)
        self.assertFalse(out["written"])
        release.assert_called_once_with("auto_brief", "scan-1")

    def test_failed_stock_note_retries_the_scan(self):
        from services import premium_auto as pa

        obs = {"run_id": 9, "previous_run_id": 8, "previous_ranked": {}, "ranked": {"AAA": {"rank": 1}}}
        with mock.patch.object(pa, "_ai_ready", return_value=True), \
                mock.patch("api.watchlist_intel.market_observation", return_value=obs), \
                mock.patch.object(pa, "_watchlists", return_value=[{"items": [{"ticker": "AAA"}]}]), \
                mock.patch("api.ai.ticker_note", side_effect=[RuntimeError("provider"), {"text": "note"}]), \
                mock.patch("db.premium_auto.prune"):
            release = mock.Mock()
            first = pa.run_notes(dt.datetime.now(UTC), claim=lambda *a: True, release=release, accounts=["p@x.com"])
            second = pa.run_notes(dt.datetime.now(UTC), claim=lambda *a: True, release=release, accounts=["p@x.com"])
        self.assertEqual((first["written"], second["written"]), (0, 1))
        release.assert_called_once_with("auto_notes", "9")

    def test_due_windows(self):
        from services import premium_auto as pa

        sun = dt.datetime(2026, 11, 8, 21, 5, tzinfo=UTC)
        self.assertEqual(pa.watchlist_email_due(sun), "2026-W46")
        self.assertIsNone(pa.watchlist_email_due(sun.replace(hour=20)))
        fri = dt.datetime(2026, 11, 6, 21, 30, tzinfo=UTC)
        self.assertEqual(pa.journal_email_due(fri), "2026-W45")
        self.assertIsNone(pa.journal_email_due(fri.replace(minute=29)))
        self.assertTrue(pa.brief_due(dt.datetime(2026, 11, 3, 15, 0, tzinfo=UTC)))     # Tue 10:00 ET
        self.assertFalse(pa.brief_due(dt.datetime(2026, 11, 7, 15, 0, tzinfo=UTC)))    # Saturday

    def test_newly_ranked_and_picks(self):
        from services import premium_auto as pa

        obs = {"previous_run_id": 1, "previous_ranked": {"AAA": {"rank": 1}},
               "ranked": {"AAA": {"rank": 1}, "BBB": {"rank": 5}, "CCC": {"rank": 2}, "DDD": {"rank": 9}}}
        self.assertEqual(pa.newly_ranked(obs), ["CCC", "BBB", "DDD"])
        self.assertEqual(pa.newly_ranked({**obs, "previous_run_id": None}), [])
        self.assertEqual(pa.pick_note_tickers(["CCC", "BBB", "DDD"], {"DDD": 3, "BBB": 1, "CCC": 1}, 2), ["DDD", "CCC"])
        self.assertEqual(pa.pick_note_tickers(["CCC"], {"ZZZ": 1}, 5), [])

    def test_run_notes_writes_once_per_scan(self):
        from services import premium_auto as pa

        obs = {"run_id": 9, "previous_run_id": 8, "previous_ranked": {}, "ranked": {"AAA": {"rank": 1}, "BBB": {"rank": 2}}}
        claims = set()

        def claim(job, key):
            if (job, key) in claims:
                return False
            claims.add((job, key))
            return True

        with mock.patch.object(pa, "_ai_ready", return_value=True), \
                mock.patch("api.watchlist_intel.market_observation", return_value=obs), \
                mock.patch.object(pa, "_watchlists", return_value=[{"items": [{"ticker": "bbb"}]}]), \
                mock.patch("api.ai.ticker_note", return_value={"text": "note"}) as note, \
                mock.patch("db.premium_auto.prune"):
            out = pa.run_notes(dt.datetime.now(UTC), claim=claim, accounts=["p@x.com"])
            again = pa.run_notes(dt.datetime.now(UTC), claim=claim, accounts=["p@x.com"])
        self.assertEqual((out["written"], out["on_watchlists"]), (1, 1))
        note.assert_called_once_with(None, "BBB")
        self.assertEqual(again, {"skipped": "done"})


class EmailTests(unittest.TestCase):
    def test_weekly_email_needs_verification_and_opt_in_before_claiming_or_generating(self):
        from services import premium_auto as pa

        for verified, opted_in in ((False, True), (True, False)):
            with self.subTest(verified=verified, opted_in=opted_in), \
                    mock.patch.object(pa, "_ai_ready", return_value=True), \
                    mock.patch("db.email_verification.is_email_verified", return_value=verified), \
                    mock.patch("db.email_prefs.wants_email", return_value=opted_in), \
                    mock.patch("db.referrals.claim_email") as claim, \
                    mock.patch("api.ai.watchlist_note") as note, \
                    mock.patch.object(pa, "_send") as send:
                out = pa.run_watchlist_email(dt.datetime(2026, 11, 8, 22, tzinfo=UTC), accounts=["p@x.com"])
            self.assertEqual(out["skipped"], {"no_email": 1})
            claim.assert_not_called()
            note.assert_not_called()
            send.assert_not_called()

    def test_paragraphs(self):
        from services.premium_auto import paragraphs

        html = paragraphs("Intro <b>.\n\n- one\n- two\nEnd")
        self.assertIn("&lt;b&gt;", html)
        self.assertEqual(html.count("<li"), 2)
        self.assertIn("End</p>", html)

    def test_compose_watchlists_and_journal(self):
        from services.premium_auto import compose_journal, compose_watchlists

        subject, text, html, _ = compose_watchlists([({"name": "Core", "items": [1, 2]}, "AAA rose."),
                                                     ({"name": "Tech", "items": [1]}, "Quiet.")], base_url="https://app")
        self.assertIn("2 watchlists", subject)
        self.assertIn("Core", html)
        self.assertIn("https://app/watchlists", text)
        subject, text, html, _ = compose_journal("- Wins beat losses.", 8, base_url="https://app")
        self.assertIn("journal", subject.lower())
        self.assertIn("8 closed trades", html)

    def test_journal_email_skips_an_unchanged_journal(self):
        from services import premium_auto as pa

        now = dt.datetime(2026, 11, 6, 22, 0, tzinfo=UTC)
        with mock.patch.object(pa, "_ai_ready", return_value=True), \
                mock.patch.object(pa, "_can_email", return_value=True), \
                mock.patch("db.referrals.claim_email", return_value=True), \
                mock.patch("db.referrals.release_email") as release, \
                mock.patch("services.lifecycle_emails._base_url", return_value="https://app"), \
                mock.patch("api.ai.stored", return_value={"text": "x", "trades_hash": "h1"}), \
                mock.patch("api.ai_premium.journal_trades_hash", return_value="h1"), \
                mock.patch("api.ai_premium.journal_coach") as coach, \
                mock.patch.object(pa, "_send") as send:
            out = pa.run_journal_email(now, accounts=["p@x.com"])
        self.assertEqual(out["skipped"], {"no_new_trades": 1})
        coach.assert_not_called()
        send.assert_not_called()
        release.assert_called_once()

    def test_watchlist_email_sends_one_card_per_list(self):
        from services import premium_auto as pa

        now = dt.datetime(2026, 11, 8, 21, 10, tzinfo=UTC)
        with mock.patch.object(pa, "_ai_ready", return_value=True), \
                mock.patch.object(pa, "_can_email", return_value=True), \
                mock.patch("db.referrals.claim_email", return_value=True), \
                mock.patch("services.lifecycle_emails._base_url", return_value="https://app"), \
                mock.patch.object(pa, "_watchlists", return_value=[{"id": 1, "name": "Core", "items": [{"ticker": "A"}]},
                                                                   {"id": 2, "name": "Tech", "items": [{"ticker": "B"}]}]), \
                mock.patch("api.ai.watchlist_note", side_effect=[{"text": "A rose."}, {"text": None}]), \
                mock.patch.object(pa, "_send", return_value=True) as send:
            out = pa.run_watchlist_email(now, accounts=["p@x.com"])
        self.assertEqual(out["sent"], 1)
        html = send.call_args[0][4]
        self.assertIn("Core", html)
        self.assertNotIn("Tech", html)


class JournalDeliveryTests(_Store):
    def test_failed_send_reuses_saved_coaching_and_retries_until_delivered(self):
        from api import ai
        from services import premium_auto as pa

        now = dt.datetime(2026, 11, 6, 22, 0, tzinfo=UTC)
        claims = set()

        def claim(email, job, key):
            entry = (email, job, key)
            if entry in claims:
                return False
            claims.add(entry)
            return True

        def coach(email):
            ai.store(email, "journal_coach", "latest", "Saved coaching", closed=6, trades_hash="h1")
            return {"text": "Saved coaching", "closed": 6}

        with mock.patch.object(pa, "_ai_ready", return_value=True), \
                mock.patch.object(pa, "_can_email", return_value=True), \
                mock.patch("db.referrals.claim_email", side_effect=claim), \
                mock.patch("db.referrals.release_email", side_effect=lambda *args: claims.discard(args)), \
                mock.patch("services.lifecycle_emails._base_url", return_value="https://app"), \
                mock.patch("api.ai_premium.journal_trades_hash", return_value="h1"), \
                mock.patch("api.ai_premium.journal_coach", side_effect=coach) as generate, \
                mock.patch.object(pa, "_send", side_effect=[False, True]) as send:
            first = pa.run_journal_email(now, accounts=["p@x.com"])
            second = pa.run_journal_email(now, accounts=["p@x.com"])
            third = pa.run_journal_email(now, accounts=["p@x.com"])
            next_week = pa.run_journal_email(now + dt.timedelta(days=7), accounts=["p@x.com"])
        self.assertEqual(first["skipped"], {"email_failed": 1})
        self.assertEqual(second["sent"], 1)
        self.assertEqual(third["skipped"], {"already_sent": 1})
        self.assertEqual(next_week["skipped"], {"no_new_trades": 1})
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(send.call_count, 2)


class ScheduledScanAITests(unittest.TestCase):
    RESULT = {"setups": [{"ticker": "AAA", "score": 88, "primary_setup": "breakout", "chg_pct": 3.2, "rvol": 2.1}],
              "symbols_scanned": 500, "label": "S&P 500", "total": 1}

    def test_summary_table(self):
        from services.scheduled_scans import summary_table

        t = summary_table(self.RESULT["setups"])
        self.assertTrue(t.startswith("ticker,score,primary_setup"))
        self.assertIn("AAA,88,Breakout", t)

    def test_compose_with_ai_text(self):
        from services.scheduled_scans import compose

        _, text, html, _ = compose("Gappers", "open", self.RESULT, base_url="https://app", ai_text="- **AAA** leads.")
        self.assertIn("AI read of these results", html)
        self.assertIn("AAA leads.", html)
        self.assertIn("AI read of these results", text)
        _, _, plain, _ = compose("Gappers", "open", self.RESULT, base_url="https://app")
        self.assertNotIn("AI read", plain)

    def test_ai_summary_off_or_empty(self):
        from services import scheduled_scans as ss

        self.assertIsNone(ss.ai_summary("p@x.com", {"setups": []}))
        with mock.patch.dict("os.environ", {"HSF_SCHEDULED_SCAN_AI": "0"}):
            self.assertIsNone(ss.ai_summary("p@x.com", self.RESULT))
        with mock.patch("ui.ai.ask_claude", return_value=("- AAA leads.", None)) as ask:
            self.assertEqual(ss.ai_summary("p@x.com", self.RESULT), "- AAA leads.")
        self.assertEqual(ask.call_args.kwargs["username"], "p@x.com")


class TradeAlertTests(_Store):
    def test_levels(self):
        from api.trade_alerts import levels

        self.assertEqual(levels(100.0, 4.0), {"stop": 96.0, "target": 106.0, "stop_pct": 4.0})
        self.assertEqual(levels(100.0, None)["stop"], 95.0)          # not in the scan: 5%
        self.assertEqual(levels(100.0, 20.0)["stop"], 92.0)          # clamped to 8%

    def _ent(self, premium=True):
        return {"entitlements": {"can_auto_research": premium}, "alert_limit": 25}

    def test_add_then_remove_when_flat(self):
        from api import trade_alerts

        made = []

        def create(user, limit, atype, **kw):
            made.append((atype, kw["direction"], kw["threshold"]))
            return {"id": len(made)}

        with mock.patch.object(trade_alerts, "_entitlements", return_value=self._ent()), \
                mock.patch.object(trade_alerts, "_stop_pct", return_value=4.0), \
                mock.patch("api.user_data.create_alert", side_effect=create):
            out = trade_alerts.add_for_trade("p@x.com", "aaa", 100.0)
            self.assertIsNone(trade_alerts.add_for_trade("p@x.com", "AAA", 101.0))   # one pair per stock
        self.assertEqual(made, [("price", "below", 96.0), ("price", "above", 106.0)])
        self.assertEqual(out["alert_ids"], [1, 2])
        self.assertEqual(trade_alerts.for_journal("p@x.com"), {"AAA": {"stop": 96.0, "target": 106.0}})

        with mock.patch("db.trades.list_trades", return_value=[{"ticker": "AAA", "closed_at": None}]), \
                mock.patch("api.user_data.delete_alert") as delete:
            self.assertEqual(trade_alerts.remove_if_flat("p@x.com", "AAA"), 0)   # another lot still open
            delete.assert_not_called()
        with mock.patch("db.trades.list_trades", return_value=[{"ticker": "AAA", "closed_at": "2026-11-03"}]), \
                mock.patch("api.user_data.delete_alert") as delete:
            self.assertEqual(trade_alerts.remove_if_flat("p@x.com", "AAA"), 2)
        self.assertEqual(trade_alerts.for_journal("p@x.com"), {})

    def test_not_premium_or_switched_off(self):
        from api import trade_alerts

        with mock.patch.object(trade_alerts, "_entitlements", return_value=self._ent(False)), \
                mock.patch("api.user_data.create_alert") as create:
            self.assertIsNone(trade_alerts.add_for_trade("p@x.com", "AAA", 100.0))
        create.assert_not_called()
        with mock.patch.dict("os.environ", {"HSF_AUTO_TRADE_ALERTS": "0"}):
            self.assertIsNone(trade_alerts.add_for_trade("p@x.com", "AAA", 100.0))

    def test_limit_reached_saves_nothing(self):
        from api import trade_alerts, user_data

        with mock.patch.object(trade_alerts, "_entitlements", return_value=self._ent()), \
                mock.patch.object(trade_alerts, "_stop_pct", return_value=None), \
                mock.patch("api.user_data.create_alert", side_effect=user_data.LimitReached("full")):
            self.assertIsNone(trade_alerts.add_for_trade("p@x.com", "AAA", 100.0))
        self.assertEqual(trade_alerts.for_journal("p@x.com"), {})

    def test_one_remaining_slot_does_not_report_a_pair_or_leave_a_partial_alert(self):
        from api import trade_alerts, user_data

        with mock.patch.object(trade_alerts, "_entitlements", return_value=self._ent()), \
                mock.patch.object(trade_alerts, "_stop_pct", return_value=None), \
                mock.patch("api.user_data.create_alert", side_effect=[{"id": 12}, user_data.LimitReached("full")]), \
                mock.patch("api.user_data.delete_alert") as delete:
            self.assertIsNone(trade_alerts.add_for_trade("p@x.com", "AAA", 100.0))
        delete.assert_called_once_with("p@x.com", 12)
        self.assertEqual(trade_alerts.for_journal("p@x.com"), {})


if __name__ == "__main__":
    unittest.main()
