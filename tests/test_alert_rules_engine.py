"""Alert rules: the pure rule check, crossing semantics, and the evaluator end to end
against the real SQL on SQLite (db.alert_rules runs the same statements on Neon).

No network, no Postgres. Market observations are built by hand in the shape
api.watchlist_intel.market_observation returns.
"""
import datetime as dt
import importlib.util
import os
import sqlite3
import unittest
from unittest import mock

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 8, 15, 0, tzinfo=UTC)


def opp(score, rank, *, signals=("breakout", "gapper"), setup="Breakout", rvol=1.2, prob=None, status="WATCH"):
    return {"score": score, "rank": rank, "signals": list(signals), "primary_setup": setup, "status": status,
            "rvol": rvol, "prob": prob, "prob_rank": None, "last": 10.0, "chg_pct": 1.0, "fading": False,
            "breakout_score": 9.0, "n_signals": len(signals)}


def observation(run_id, ranked, raw=None, prev=None, scan_at=None):
    raw = raw or {}
    for t in ranked:
        raw.setdefault(t, {"last": 10.0, "chg_pct": 1.0, "rvol": ranked[t].get("rvol"), "ema_cross": None})
    return {"observation_id": f"run:{run_id}", "run_id": run_id, "scan_at": scan_at or NOW - dt.timedelta(minutes=5),
            "previous_run_id": None, "previous_scan_at": None, "ranked": ranked, "total": len(ranked),
            "previous_ranked": prev or {}, "previous_rows": None, "raw": raw}


class RuleCheckTests(unittest.TestCase):
    def setUp(self):
        from api.alert_rules import check

        self.check = check

    def view(self, **kw):
        base = {"present": True, "ranked": True, "hsf_score": None, "rank": None, "setup": None,
                "prebreakout": None, "prebreakout_score": None, "rvol": None}
        return {**base, **kw}

    def run_seq(self, rule, views):
        state, fired = None, []
        for i, v in enumerate(views):
            f, value, prev, new = self.check(rule, v, state)
            fired.append(f)
            if new is not None:
                state = {**(state or {}), **new, "observation_id": f"run:{i}"}
        return fired

    def test_cross_above_fires_on_the_transition_only(self):
        rule = {"rule_type": "HSF_SCORE_CROSS_ABOVE", "threshold": 80.0}
        scores = [77, 82, 83, 84, 79, 81]
        fired = self.run_seq(rule, [self.view(hsf_score=s) for s in scores])
        self.assertEqual(fired, [False, True, False, False, False, True])

    def test_first_observation_is_only_a_baseline_for_crossings(self):
        rule = {"rule_type": "HSF_SCORE_CROSS_ABOVE", "threshold": 80.0}
        self.assertEqual(self.run_seq(rule, [self.view(hsf_score=83), self.view(hsf_score=84)]), [False, False])

    def test_cross_below(self):
        rule = {"rule_type": "HSF_SCORE_CROSS_BELOW", "threshold": 60.0}
        fired = self.run_seq(rule, [self.view(hsf_score=s) for s in (65, 59, 55, 61, 58)])
        self.assertEqual(fired, [False, True, False, False, True])

    def test_unranked_keeps_last_known_score(self):
        rule = {"rule_type": "HSF_SCORE_CROSS_ABOVE", "threshold": 80.0}
        views = [self.view(hsf_score=77), self.view(ranked=False, hsf_score=None), self.view(hsf_score=82)]
        self.assertEqual(self.run_seq(rule, views), [False, False, True])

    def test_absent_from_scan_changes_nothing(self):
        f, _, _, new = self.check({"rule_type": "HSF_SCORE_ABOVE", "threshold": 80.0}, {"present": False}, None)
        self.assertFalse(f)
        self.assertIsNone(new)

    def test_level_rules(self):
        above = {"rule_type": "HSF_SCORE_ABOVE", "threshold": 80.0}
        below = {"rule_type": "HSF_SCORE_BELOW", "threshold": 60.0}
        self.assertEqual(self.run_seq(above, [self.view(hsf_score=s) for s in (79, 80, 90)]), [False, True, True])
        self.assertEqual(self.run_seq(below, [self.view(hsf_score=s) for s in (60, 59)]), [False, True])
        # no HSF score is not "below 60"
        self.assertEqual(self.run_seq(below, [self.view(ranked=False)]), [False])

    def test_rvol(self):
        rule = {"rule_type": "RVOL_ABOVE", "threshold": 2.0}
        self.assertEqual(self.run_seq(rule, [self.view(rvol=v) for v in (1.5, 2.0, None)]), [False, True, False])

    def test_prebreakout_activation(self):
        rule = {"rule_type": "PREBREAKOUT_ACTIVE", "threshold": None}
        views = [self.view(prebreakout=False), self.view(prebreakout=True), self.view(prebreakout=True),
                 self.view(ranked=False, prebreakout=False), self.view(prebreakout=True)]
        self.assertEqual(self.run_seq(rule, views), [False, True, False, False, True])
        # already active when the rule first sees it: baseline, not an alert
        self.assertEqual(self.run_seq(rule, [self.view(prebreakout=True)]), [False])

    def test_setup_appeared_with_and_without_a_setup_filter(self):
        any_setup = {"rule_type": "SETUP_APPEARED", "threshold": None, "value": None}
        breakout = {"rule_type": "SETUP_APPEARED", "threshold": None, "value": "breakout"}
        views = [self.view(ranked=False), self.view(setup="Golden Cross"), self.view(setup="Breakout")]
        self.assertEqual(self.run_seq(any_setup, views), [False, True, False])
        self.assertEqual(self.run_seq(breakout, views), [False, False, True])

    def test_rank_improved(self):
        rule = {"rule_type": "RANK_IMPROVED", "threshold": 10.0}
        views = [self.view(rank=40), self.view(rank=35), self.view(rank=20), self.view(ranked=False),
                 self.view(rank=5)]
        self.assertEqual(self.run_seq(rule, views), [False, False, True, False, False])


class Store:
    """One in-memory SQLite database with the app tables the evaluator reads.
    HSF_TEST_PG_URL=postgresql://... runs the same tests on a scratch Postgres instead
    (its tables are dropped and recreated)."""

    def __init__(self):
        from db import alert_rules

        url = os.environ.get("HSF_TEST_PG_URL", "").strip()
        self.pg = bool(url)
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row

            self.conn = psycopg.connect(url, row_factory=dict_row)
            for t in ("users", "watchlists", "watchlist_items", "user_alerts", "hsf_alert_rules",
                      "hsf_alert_rule_state", "hsf_alert_rule_events"):
                self.conn.execute(f"DROP TABLE IF EXISTS {t}")
            serial = "SERIAL PRIMARY KEY"
        else:
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            serial = "INTEGER PRIMARY KEY"
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE users (username TEXT, tier TEXT, is_admin BOOLEAN, email_verified BOOLEAN, "
                    "is_active BOOLEAN)")
        cur.execute("CREATE TABLE watchlists (id INTEGER PRIMARY KEY, user_id TEXT, name TEXT)")
        cur.execute(f"CREATE TABLE watchlist_items (id {serial}, watchlist_id INTEGER, ticker TEXT)")
        cur.execute(f"CREATE TABLE user_alerts (id {serial}, user_id TEXT, enabled BOOLEAN, ticker TEXT)")
        self.conn.commit()
        if self.pg:
            from db.engine import _WarmConn  # close() ends the transaction, like production's warm connection

            warm = _WarmConn(self.conn)
            alert_rules.set_connection_factory(lambda: warm)
        else:
            alert_rules.set_connection_factory(lambda: self.conn)
        alert_rules._ensure_schema.__wrapped__(self.conn, not self.pg)  # schema_once is per database key

    def x(self, sql, args=()):
        self.conn.execute(sql.replace("?", "%s") if self.pg else sql, args)
        self.conn.commit()

    def user(self, name, tier="pro", verified=True, active=True, admin=False):
        self.x("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (name, tier, admin, verified, active))

    def watchlist(self, wid, user, tickers):
        self.x("INSERT INTO watchlists (id, user_id, name) VALUES (?, ?, ?)", (wid, user, f"W{wid}"))
        for t in tickers:
            self.x("INSERT INTO watchlist_items (watchlist_id, ticker) VALUES (?, ?)", (wid, t))

    def legacy_alerts(self, user, n):
        for _ in range(n):
            self.x("INSERT INTO user_alerts (user_id, enabled, ticker) VALUES (?, TRUE, 'X')", (user,))

    def events(self):
        import json as _json

        rows = [dict(r) for r in self.conn.execute("SELECT * FROM hsf_alert_rule_events ORDER BY id").fetchall()]
        for r in rows:  # same shapes as SQLite for the assertions
            if not isinstance(r["delivery"], str):
                r["delivery"] = _json.dumps(r["delivery"])
        return rows

    def close(self):
        from db import alert_rules

        alert_rules.set_connection_factory(None)
        self.conn.close()


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "needs fastapi (entitlements_for lives in api.main)")
class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        from api import alert_rules
        from db import alert_rules as store

        self.ar, self.store = alert_rules, store
        self.db = Store()
        self.addCleanup(self.db.close)
        self.db.user("a@x.com", "pro")
        self.db.user("b@x.com", "basic")
        self.db.user("p@x.com", "premium")
        self.emails = []
        self.email_ok = True
        mock.patch("api.alert_rules._send_email", side_effect=self._send).start()
        mock.patch("api.watchlist_intel.observation_stale", return_value=False).start()
        self.addCleanup(mock.patch.stopall)

    def _send(self, user, message):
        self.emails.append((user, message))
        return (True, None) if self.email_ok else (False, "smtp_failed")

    def rule(self, user, rule_type, *, ticker=None, watchlist_id=None, threshold=None, value=None,
             channels=("in_app",), cooldown=None):
        spec = self.ar.RULE_TYPES[rule_type]
        return self.store.create_rule(user, rule_type=rule_type, operator=spec["operator"], threshold=threshold,
                                      value=value, ticker=ticker, watchlist_id=watchlist_id,
                                      delivery_channels=list(channels),
                                      cooldown_seconds=self.ar.DEFAULT_COOLDOWN[spec["kind"]] if cooldown is None else cooldown)

    def ev(self, obs, now=NOW):
        return self.ar.evaluate_once(now, observation=obs)

    def test_crossing_fires_once_and_reevaluation_is_idempotent(self):
        self.rule("a@x.com", "HSF_SCORE_CROSS_ABOVE", ticker="MXL", threshold=80)
        self.assertEqual(self.ev(observation(1, {"MXL": opp(77, 3)}))["triggered"], 0)
        r = self.ev(observation(2, {"MXL": opp(82, 2)}), NOW + dt.timedelta(minutes=30))
        self.assertEqual(r["triggered"], 1)
        again = self.ev(observation(2, {"MXL": opp(82, 2)}), NOW + dt.timedelta(minutes=31))
        self.assertEqual((again["triggered"], again["evaluated"]), (0, 0))  # same scan: nothing to do
        r = self.ev(observation(3, {"MXL": opp(84, 2)}), NOW + dt.timedelta(hours=3))
        self.assertEqual(r["triggered"], 0)  # still above: not a new crossing
        evs = self.db.events()
        self.assertEqual(len(evs), 1)
        e = evs[0]
        self.assertEqual((e["ticker"], e["trigger_value"], e["previous_value"], e["threshold"], e["observation_id"]),
                         ("MXL", 82.0, 77.0, 80.0, "run:2"))
        self.assertIn("crossed above 80", e["message"])
        rule = self.store.list_rules("a@x.com")[0]
        self.assertIsNotNone(rule["last_triggered_at"])
        self.assertIsNotNone(rule["last_evaluated_at"])

    def test_duplicate_insert_is_ignored_by_the_unique_key(self):
        rule = self.rule("a@x.com", "HSF_SCORE_ABOVE", ticker="MXL", threshold=80)
        ev = {"user_id": "a@x.com", "rule_id": rule["id"], "watchlist_id": None, "ticker": "MXL",
              "rule_type": "HSF_SCORE_ABOVE", "operator": ">=", "threshold": 80.0, "trigger_value": 85.0,
              "previous_value": None, "hsf_score": 85.0, "setup": "Breakout", "message": "m",
              "observation_id": "run:9", "market_data_as_of": None, "triggered_at": NOW.isoformat(),
              "delivery": {"in_app": "delivered"}}
        first = self.store.save_evaluation(states=[], events=[ev], evaluated_rule_ids=[], triggered_rule_ids=[],
                                           now=NOW.isoformat())
        second = self.store.save_evaluation(states=[], events=[ev], evaluated_rule_ids=[], triggered_rule_ids=[],
                                            now=NOW.isoformat())
        self.assertEqual((len(first), len(second), len(self.db.events())), (1, 0, 1))

    def test_level_rule_respects_cooldown(self):
        self.rule("a@x.com", "HSF_SCORE_ABOVE", ticker="MXL", threshold=80, cooldown=3600)
        self.assertEqual(self.ev(observation(1, {"MXL": opp(85, 1)}))["triggered"], 1)
        r = self.ev(observation(2, {"MXL": opp(86, 1)}), NOW + dt.timedelta(minutes=30))
        self.assertEqual((r["triggered"], r["deduplicated"]), (0, 1))
        self.assertEqual(self.ev(observation(3, {"MXL": opp(87, 1)}), NOW + dt.timedelta(hours=2))["triggered"], 1)

    def test_watchlist_rule_covers_every_symbol_and_multiple_users(self):
        self.db.watchlist(1, "a@x.com", ["MXL", "STM", "ZZZZ"])
        self.db.watchlist(2, "b@x.com", ["STM"])
        self.rule("a@x.com", "RVOL_ABOVE", watchlist_id=1, threshold=2.0)
        self.rule("b@x.com", "RVOL_ABOVE", watchlist_id=2, threshold=2.0)
        self.rule("b@x.com", "RVOL_ABOVE", watchlist_id=1, threshold=3.0)  # not b's list: ignored
        obs = observation(1, {"MXL": opp(70, 1, rvol=2.5), "STM": opp(60, 2, rvol=4.0)})
        r = self.ev(obs)
        got = sorted((e["user_id"], e["ticker"]) for e in self.db.events())
        self.assertEqual(got, [("a@x.com", "MXL"), ("a@x.com", "STM"), ("b@x.com", "STM")])
        self.assertEqual(r["triggered"], 3)

    def test_deleted_watchlist_rules_are_skipped(self):
        self.rule("a@x.com", "RVOL_ABOVE", watchlist_id=99, threshold=2.0)
        r = self.ev(observation(1, {"MXL": opp(70, 1, rvol=5.0)}))
        self.assertEqual((r["triggered"], r["evaluated"]), (0, 0))

    def test_disabled_rules_are_not_evaluated(self):
        rule = self.rule("a@x.com", "RVOL_ABOVE", ticker="MXL", threshold=2.0)
        self.store.update_rule("a@x.com", rule["id"], {"enabled": False})
        self.assertEqual(self.ev(observation(1, {"MXL": opp(70, 1, rvol=5.0)}))["skipped"], "no_rules")

    def test_stale_or_missing_scan_skips_without_touching_state(self):
        self.rule("a@x.com", "HSF_SCORE_CROSS_ABOVE", ticker="MXL", threshold=80)
        self.ev(observation(1, {"MXL": opp(77, 1)}))
        with mock.patch("api.watchlist_intel.observation_stale", return_value=True):
            self.assertEqual(self.ev(observation(2, {"MXL": opp(78, 1)}))["skipped"], "stale_scan")
        with mock.patch("api.watchlist_intel.market_observation", return_value=None):
            self.assertEqual(self.ar.evaluate_once(NOW)["skipped"], "no_scan")
        # the crossing is still caught on the next fresh scan, measured from the last seen 77
        self.assertEqual(self.ev(observation(3, {"MXL": opp(81, 1)}))["triggered"], 1)

    def test_prebreakout_rules_need_premium_now(self):
        self.rule("p@x.com", "PREBREAKOUT_ACTIVE", ticker="MXL")
        self.ev(observation(1, {"MXL": opp(70, 1, signals=("breakout", "gapper"))}))
        r = self.ev(observation(2, {"MXL": opp(75, 1, signals=("breakout", "prebreakout"), prob=40.0)}))
        self.assertEqual(r["triggered"], 1)
        self.db.x("UPDATE users SET tier = 'pro' WHERE username = 'p@x.com'")  # downgraded
        self.ev(observation(3, {"MXL": opp(70, 1)}))
        r = self.ev(observation(4, {"MXL": opp(75, 1, signals=("breakout", "prebreakout"), prob=40.0)}))
        self.assertEqual(r["rules"], 0)

    def test_setup_events_are_redacted_below_premium(self):
        self.rule("a@x.com", "SETUP_APPEARED", ticker="MXL")
        self.ev(observation(1, {}, raw={"MXL": {"last": 9.0, "chg_pct": 0.1, "rvol": 1.0, "ema_cross": None}}))
        self.ev(observation(2, {"MXL": opp(66, 4, signals=("prebreakout", "gapper"), setup="PreBreakout", prob=50)}))
        e = self.db.events()[0]
        self.assertNotIn("prebreakout", (e["setup"] or "").lower())
        self.assertNotIn("prebreakout", e["message"].lower())

    def test_plan_limit_counts_ticker_alerts_and_newest_rules_win(self):
        self.db.legacy_alerts("b@x.com", 0)
        self.rule("b@x.com", "RVOL_ABOVE", ticker="MXL", threshold=2.0)
        # Free = 1 active alert; a second rule made while over (e.g. via a downgrade) is not evaluated
        self.store.create_rule("b@x.com", rule_type="RVOL_ABOVE", operator=">=", threshold=2.0, value=None,
                               ticker="STM", watchlist_id=None, delivery_channels=["in_app"], cooldown_seconds=0)
        self.ev(observation(1, {"MXL": opp(70, 1, rvol=5.0), "STM": opp(60, 2, rvol=5.0)}))
        self.assertEqual([e["ticker"] for e in self.db.events()], ["STM"])  # newest only
        with self.assertRaises(self.store.RuleLimitReached):
            self.store.create_rule("b@x.com", rule_type="RVOL_ABOVE", operator=">=", threshold=3.0, value=None,
                                   ticker="GRAL", watchlist_id=None, delivery_channels=["in_app"],
                                   cooldown_seconds=0, max_active=1)
        self.db.user("c@x.com", "basic")
        self.db.legacy_alerts("c@x.com", 1)
        with self.assertRaises(self.store.RuleLimitReached):  # the ticker alert already uses the one slot
            self.store.create_rule("c@x.com", rule_type="RVOL_ABOVE", operator=">=", threshold=3.0, value=None,
                                   ticker="GRAL", watchlist_id=None, delivery_channels=["in_app"],
                                   cooldown_seconds=0, max_active=1)

    def test_email_delivery_and_failure_keeps_the_event_for_retry(self):
        self.rule("a@x.com", "RVOL_ABOVE", ticker="MXL", threshold=2.0, channels=("in_app", "email"), cooldown=0)
        self.email_ok = False
        r = self.ev(observation(1, {"MXL": opp(70, 1, rvol=3.0)}))
        self.assertEqual((r["triggered"], r["delivery_failed"]), (1, 1))
        import json
        e = self.db.events()[0]
        self.assertEqual(json.loads(e["delivery"]), {"in_app": "delivered", "email": "failed"})
        self.assertEqual(e["delivery_attempts"], 1)
        self.email_ok = True
        ok, failed = self.ar.retry_failed_deliveries(NOW + dt.timedelta(minutes=1))
        self.assertEqual((ok, failed), (1, 0))
        e = self.db.events()[0]
        self.assertEqual(json.loads(e["delivery"])["email"], "sent")
        self.assertEqual(self.ar.retry_failed_deliveries(NOW + dt.timedelta(minutes=2)), (0, 0))

    def test_email_channel_dropped_after_downgrade_and_unverified_skipped(self):
        self.rule("a@x.com", "RVOL_ABOVE", ticker="MXL", threshold=2.0, channels=("in_app", "email"))
        self.db.x("UPDATE users SET email_verified = FALSE WHERE username = 'a@x.com'")
        self.ev(observation(1, {"MXL": opp(70, 1, rvol=3.0)}))
        import json
        self.assertEqual(json.loads(self.db.events()[0]["delivery"])["email"], "skipped")
        self.assertEqual(self.emails, [])

    def test_delivery_status_failure_does_not_lose_the_event(self):
        self.rule("a@x.com", "RVOL_ABOVE", ticker="MXL", threshold=2.0, channels=("in_app", "email"))
        with mock.patch("db.alert_rules.set_delivery", side_effect=RuntimeError("db blip")):
            r = self.ev(observation(1, {"MXL": opp(70, 1, rvol=3.0)}))
        self.assertEqual(r["triggered"], 1)
        self.assertEqual(len(self.db.events()), 1)

    def test_rank_improvement_and_metrics(self):
        self.rule("a@x.com", "RANK_IMPROVED", ticker="MXL", threshold=5)
        before = self.ar.metrics()["rules_triggered"]
        self.ev(observation(1, {"MXL": opp(60, 30)}))
        self.ev(observation(2, {"MXL": opp(70, 12)}))
        self.assertIn("moved up 18 places to #12", self.db.events()[0]["message"])
        self.assertEqual(self.ar.metrics()["rules_triggered"] - before, 1)

    def test_changing_the_threshold_resets_the_baseline(self):
        rule = self.rule("a@x.com", "HSF_SCORE_CROSS_ABOVE", ticker="MXL", threshold=80)
        self.ev(observation(1, {"MXL": opp(77, 1)}))
        self.store.update_rule("a@x.com", rule["id"], {"threshold": 90.0})
        self.assertEqual(self.ev(observation(2, {"MXL": opp(95, 1)}))["triggered"], 0)  # new baseline
        self.assertEqual(self.store.load_states([rule["id"]])[(rule["id"], "MXL")]["last_value"], 95.0)

    def test_worker_pass_never_raises_and_respects_the_kill_switch(self):
        with mock.patch("api.alert_rules.evaluate_once", side_effect=RuntimeError("boom")) as ev, \
                mock.patch("billing_service.realtime_alerts.market_session_open", return_value=True) as session:
            self.ar.worker_pass()
            self.assertEqual(ev.call_count, 1)
            with mock.patch.dict("os.environ", {"HSF_ALERT_RULES_ENABLED": "0"}):
                self.ar.worker_pass()
            self.assertEqual(ev.call_count, 1)
            session.return_value = False  # overnight / weekends: no database work
            self.ar.worker_pass()
            self.assertEqual(ev.call_count, 1)

    def test_worker_loop_runs_registered_hooks(self):
        from billing_service import realtime_alerts as rt

        calls = []
        hook = lambda: calls.append(1)  # noqa: E731
        rt.register_pass_hook(hook)
        self.addCleanup(rt._pass_hooks.remove, hook)
        rt.register_pass_hook(hook)  # once only
        rt.run_hooks()
        self.assertEqual(calls, [1])


class ValidationTests(unittest.TestCase):
    def test_rules_validation_and_entitlements(self):
        from api.alert_rules import RuleError, RuleForbidden, validate

        free = {"can_early_breakout": False, "can_email_alerts": False}
        prem = {"can_early_breakout": True, "can_email_alerts": True}
        ok = validate(free, rule_type="hsf_score_cross_above", threshold=80, value=None, ticker="mxl",
                      watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        self.assertEqual((ok["rule_type"], ok["ticker"], ok["operator"], ok["delivery_channels"],
                          ok["cooldown_seconds"]), ("HSF_SCORE_CROSS_ABOVE", "MXL", "crosses_above", ["in_app"], 3600))
        bad = [dict(rule_type="NOPE"), dict(threshold=None), dict(threshold=101), dict(ticker="!!"),
               dict(ticker=None), dict(watchlist_id=3), dict(delivery_channels=["sms"]),
               dict(cooldown_seconds=-1), dict(value="x")]
        for b in bad:
            args = dict(rule_type="HSF_SCORE_ABOVE", threshold=80, value=None, ticker="MXL", watchlist_id=None,
                        delivery_channels=None, cooldown_seconds=None)
            args.update(b)
            with self.assertRaises(RuleError, msg=str(b)):
                validate(prem, **args)
        with self.assertRaises(RuleForbidden):
            validate(free, rule_type="PREBREAKOUT_ACTIVE", threshold=None, value=None, ticker="MXL",
                     watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        with self.assertRaises(RuleForbidden):
            validate(free, rule_type="RVOL_ABOVE", threshold=2, value=None, ticker="MXL", watchlist_id=None,
                     delivery_channels=["email"], cooldown_seconds=None)
        with self.assertRaises(RuleError):
            validate(prem, rule_type="PREBREAKOUT_ACTIVE", threshold=5, value=None, ticker="MXL",
                     watchlist_id=None, delivery_channels=None, cooldown_seconds=None)


if __name__ == "__main__":
    unittest.main()
