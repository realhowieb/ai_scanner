"""Tests for the real-time price-alert worker's pure logic."""
import datetime as dt
import unittest

from billing_service.realtime_alerts import crossed, market_session_open


def _utc(iso: str) -> dt.datetime:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc)


class CrossedTests(unittest.TestCase):
    def test_above(self):
        self.assertTrue(crossed("above", 401.0, 400.0))
        self.assertTrue(crossed("above", 400.0, 400.0))
        self.assertFalse(crossed("above", 399.99, 400.0))

    def test_below(self):
        self.assertTrue(crossed("below", 399.0, 400.0))
        self.assertFalse(crossed("below", 400.01, 400.0))

    def test_unknown_direction_defaults_to_above(self):
        self.assertTrue(crossed("sideways", 401.0, 400.0))


class MarketSessionTests(unittest.TestCase):
    # July = EDT (UTC-4). Monday 2026-07-13.
    def test_extended_hours_open(self):
        self.assertTrue(market_session_open(_utc("2026-07-13T08:05:00")))   # 4:05a ET premarket
        self.assertTrue(market_session_open(_utc("2026-07-13T14:00:00")))   # 10:00a ET
        self.assertTrue(market_session_open(_utc("2026-07-13T23:59:00")))   # 7:59p ET afterhours

    def test_overnight_closed(self):
        self.assertFalse(market_session_open(_utc("2026-07-14T00:05:00")))  # 8:05p ET Mon
        self.assertFalse(market_session_open(_utc("2026-07-13T07:55:00")))  # 3:55a ET

    def test_weekend_closed(self):
        self.assertFalse(market_session_open(_utc("2026-07-11T14:00:00")))  # Saturday


class EntitlementLimitTests(unittest.TestCase):
    def test_realtime_worker_enforces_plan_limits_after_downgrade(self):
        from billing_service.realtime_alerts import _apply_plan_limits

        alerts = [
            {"id": i, "user_id": "free@example.com", "tier": "basic"}
            for i in range(3)
        ] + [
            {"id": i, "user_id": "pro@example.com", "tier": "pro"}
            for i in range(10)
        ]
        kept = _apply_plan_limits(alerts)
        self.assertEqual([a["id"] for a in kept if a["user_id"].startswith("free")], [0])
        self.assertEqual(len([a for a in kept if a["user_id"].startswith("pro")]), 5)


if __name__ == "__main__":
    unittest.main()


class EvaluateAlertTests(unittest.TestCase):
    def _alert(self, atype, threshold, direction="above"):
        return {
            "ticker": "TSLA",
            "alert_type": atype,
            "threshold": threshold,
            "direction": direction,
        }

    def test_price_alert_fires(self):
        from billing_service.realtime_alerts import evaluate_alert

        msg = evaluate_alert(self._alert("price", 400.0), {"last": 401.0})
        self.assertIn("above your 400.00", msg)
        self.assertIsNone(evaluate_alert(self._alert("price", 400.0), {"last": 399.0}))

    def test_move_alert_fires_both_directions(self):
        from billing_service.realtime_alerts import evaluate_alert

        up = evaluate_alert(self._alert("move", 5.0), {"last": 106.0, "prev_close": 100.0})
        self.assertIn("+6.0%", up)
        down = evaluate_alert(self._alert("move", 5.0), {"last": 94.0, "prev_close": 100.0})
        self.assertIn("-6.0%", down)
        self.assertIsNone(
            evaluate_alert(self._alert("move", 5.0), {"last": 103.0, "prev_close": 100.0})
        )

    def test_move_without_prev_close_is_silent(self):
        from billing_service.realtime_alerts import evaluate_alert

        self.assertIsNone(evaluate_alert(self._alert("move", 5.0), {"last": 106.0}))

    def test_monday_premarket_does_not_realert_fridays_move(self):
        """2026-10-12 04:00 ET: NVAX's snapshot still held Friday (+15.9% vs Thursday),
        and the worker emailed it as "today". No trade yet today means no alert."""
        import datetime as dt
        from unittest import mock

        from billing_service import realtime_alerts as ra

        payload = {"NVAX": {
            "latestTrade": {"t": "2026-10-09T23:59:00Z", "p": 12.74},
            "minuteBar": {"t": "2026-10-09T23:59:00Z", "c": 12.74, "v": 100},
            "dailyBar": {"t": "2026-10-09T04:00:00Z", "c": 12.70, "v": 9_000_000},
            "prevDailyBar": {"t": "2026-10-08T04:00:00Z", "c": 10.99, "v": 4_000_000},
        }}
        resp = mock.Mock(json=mock.Mock(return_value=payload), raise_for_status=mock.Mock())
        fake_httpx = mock.Mock(get=mock.Mock(return_value=resp))  # smoke CI has no httpx
        with mock.patch.dict("os.environ", {"ALPACA_API_KEY_ID": "k", "ALPACA_API_SECRET_KEY": "s"}), \
                mock.patch.dict("sys.modules", {"httpx": fake_httpx}):
            snap = ra._latest_snapshots(["NVAX"])["NVAX"]
        monday = ra._today_et(dt.datetime(2026, 10, 12, 8, 0, tzinfo=dt.timezone.utc))
        alert = {**self._alert("move", 5.0), "ticker": "NVAX"}
        self.assertIsNone(ra.evaluate_alert(alert, snap, today=monday))
        self.assertIsNone(ra.evaluate_alert({**alert, "alert_type": "rvol", "threshold": 1.0},
                                            snap, today=monday))
        self.assertIsNone(ra.evaluate_alert({**alert, "alert_type": "price", "threshold": 10.0},
                                            snap, today=monday))
        # Friday itself was a real move.
        self.assertIn("+15.9%", ra.evaluate_alert(alert, snap, today="2026-10-09"))

    def test_premarket_move_measured_from_last_close(self):
        from billing_service.realtime_alerts import evaluate_alert

        snap = {"last": 13.50, "prev_close": 10.99, "daily_close": 12.70,
                "trade_day": "2026-10-12", "daily_day": "2026-10-09"}
        msg = evaluate_alert({**self._alert("move", 5.0), "ticker": "NVAX"}, snap, today="2026-10-12")
        self.assertIn("+6.3%", msg)  # vs Friday's 12.70 close, not Thursday's 10.99
        snap["last"] = 12.90
        self.assertIsNone(evaluate_alert({**self._alert("move", 5.0)}, snap, today="2026-10-12"))

    def test_rvol_alert_uses_cached_average(self):
        import datetime as dt

        from billing_service import realtime_alerts as ra

        ra._AVG_VOL["TSLA"] = (dt.datetime.now(dt.timezone.utc).date(), 1_000_000.0)
        msg = ra.evaluate_alert(
            self._alert("rvol", 2.0), {"last": 100.0, "volume": 2_500_000.0}
        )
        self.assertIn("2.5x", msg)
        self.assertIsNone(
            ra.evaluate_alert(self._alert("rvol", 2.0), {"last": 100.0, "volume": 500_000.0})
        )
