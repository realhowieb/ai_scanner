"""Unit tests for the pre-open morning digest composition (pandas-free)."""
import unittest
from unittest import mock

from scheduler.morning_digest import (
    _compose,
    _movers_table,
    _movers_text,
    _track_record_line,
)


def _row(ticker="AAPL", last=100.0, chg=1.5, gap=0.5):
    return {"ticker": ticker, "last": last, "chg_pct": chg, "gap_pct": gap}


class MoversFormattingTests(unittest.TestCase):
    def test_movers_table_renders_rows_and_colors(self):
        html = _movers_table([_row(chg=2.0), _row(ticker="TSLA", chg=-3.0)])
        self.assertIn("AAPL", html)
        self.assertIn("TSLA", html)
        self.assertIn("+2.00%", html)
        self.assertIn("-3.00%", html)
        self.assertIn("#16a34a", html)  # green for positive
        self.assertIn("#dc2626", html)  # red for negative

    def test_movers_table_tolerates_none_values(self):
        html = _movers_table([_row(chg=None, gap=None)])
        self.assertIn("—", html)

    def test_movers_table_empty(self):
        self.assertIn("No data", _movers_table([]))

    def test_movers_text_lines(self):
        text = _movers_text([_row(), _row(ticker="MSFT", chg=None)])
        self.assertIn("AAPL", text)
        self.assertIn("+1.50%", text)
        self.assertIn("MSFT", text)
        self.assertIn("—", text)

    def test_movers_text_empty(self):
        self.assertIn("(none)", _movers_text([]))


class ComposeTests(unittest.TestCase):
    def test_compose_includes_all_sections_when_data_present(self):
        with mock.patch(
            "scheduler.morning_digest._track_record_line",
            return_value=("<p>TR</p>", "TR"),
        ):
            html, text = _compose(
                "user@example.com",
                watch_rows=[_row()],
                gappers=[_row(ticker="NVDA", chg=4.0, gap=3.0)],
                earnings_hits=["AAPL"],
                pick={"symbol": "ACLX", "prob": 93.5},
            )
        for fragment in ("Your watchlist", "Top market gappers", "Earnings today", "PreBreakout pick"):
            self.assertIn(fragment, html)
        self.assertIn("ACLX", html)
        self.assertIn("93.5", html)
        self.assertIn("TR", html)
        # Text fallback mirrors the sections.
        self.assertIn("Your watchlist", text)
        self.assertIn("ACLX", text)

    def test_compose_omits_conditional_sections(self):
        with mock.patch(
            "scheduler.morning_digest._track_record_line", return_value=("", "")
        ):
            html, text = _compose(
                "user@example.com",
                watch_rows=[_row()],
                gappers=[],
                earnings_hits=[],
                pick=None,
            )
        self.assertNotIn("Earnings today", html)
        self.assertNotIn("PreBreakout pick", html)
        self.assertNotIn("Track record", html)


class TrackRecordLineTests(unittest.TestCase):
    def _tr(self, sample_size=200, runs_used=10, avg=0.012, win=0.55):
        return {
            "horizon_days": 5,
            "avg_return": avg,
            "win_rate": win,
            "sample_size": sample_size,
            "runs_used": runs_used,
            "benchmark": "SPY",
            "top_n": 5,
        }

    def test_shows_when_sample_is_meaningful(self):
        with mock.patch(
            "db.track_record.load_latest_track_record", return_value=self._tr()
        ):
            html, text = _track_record_line()
        self.assertIn("beat the SPY", html)
        self.assertIn("+1.2%", html)
        self.assertIn("top-5", html)
        self.assertIn("SPY", text)

    def test_hidden_below_sample_gate(self):
        with mock.patch(
            "db.track_record.load_latest_track_record",
            return_value=self._tr(sample_size=60),
        ):
            self.assertEqual(_track_record_line(), ("", ""))

    def test_hidden_below_runs_gate(self):
        with mock.patch(
            "db.track_record.load_latest_track_record",
            return_value=self._tr(runs_used=3),
        ):
            self.assertEqual(_track_record_line(), ("", ""))

    def test_hidden_when_unavailable(self):
        with mock.patch(
            "db.track_record.load_latest_track_record", return_value=None
        ):
            self.assertEqual(_track_record_line(), ("", ""))

    def test_hidden_when_db_raises(self):
        with mock.patch(
            "db.track_record.load_latest_track_record",
            side_effect=RuntimeError("neon down"),
        ):
            self.assertEqual(_track_record_line(), ("", ""))


if __name__ == "__main__":
    unittest.main()


class EarningsRailTests(unittest.TestCase):
    def _emap(self, days_from_today):
        import datetime as dt

        today = dt.datetime.now(dt.timezone.utc).date()
        return {
            sym: (today + dt.timedelta(days=d)) if d is not None else None
            for sym, d in days_from_today.items()
        }

    def test_days_map_flags_only_imminent(self):
        from scheduler.morning_digest import _earnings_days_map

        with mock.patch(
            "db.earnings.load_earnings_map",
            return_value=self._emap({"SOON": 2, "LATER": 10, "NONE": None}),
        ):
            out = _earnings_days_map(["SOON", "LATER", "NONE"])
        self.assertEqual(out, {"SOON": 2})

    def test_days_map_empty_on_failure(self):
        from scheduler.morning_digest import _earnings_days_map

        with mock.patch(
            "db.earnings.load_earnings_map", side_effect=RuntimeError("db down")
        ):
            self.assertEqual(_earnings_days_map(["AAPL"]), {})

    def test_flag_rows_annotates_ticker_in_place(self):
        from scheduler.morning_digest import _flag_earnings_rows

        rows = [{"ticker": "DAL"}, {"ticker": "PEP"}]
        _flag_earnings_rows(rows, {"DAL": 2})
        self.assertEqual(rows[0]["ticker"], "DAL ⚠️E2d")
        self.assertEqual(rows[1]["ticker"], "PEP")

    def test_alert_lines_annotated(self):
        from scheduler.alert_runner import _annotate_earnings

        with mock.patch(
            "db.earnings.load_earnings_map",
            return_value=self._emap({"TSLA": 3, "AAPL": 30}),
        ):
            out = _annotate_earnings(["TSLA: crossed above 400", "AAPL: breakout 9.1"])
        self.assertIn("⚠️ earnings in 3d", out[0])
        self.assertNotIn("earnings", out[1])

    def test_alert_lines_unchanged_on_failure(self):
        from scheduler.alert_runner import _annotate_earnings

        lines = ["TSLA: crossed above 400"]
        with mock.patch(
            "db.earnings.load_earnings_map", side_effect=RuntimeError("db down")
        ):
            self.assertEqual(_annotate_earnings(lines), lines)
