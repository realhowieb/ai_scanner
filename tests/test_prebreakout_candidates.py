"""PreBreakout scores the scan's near-setup names (its training population) and the
tab, Brief and alerts read those instead of the scan's top setups."""
import importlib.util
import json
import unittest
from unittest import mock

_PANDAS = importlib.util.find_spec("pandas") is not None


def _cand(ticker, raw, *, score=5.0, last=20.0, rvol=1.2):
    return {"Ticker": ticker, "BreakoutScore": score, "Last": last, "VolRel20": rvol, "PctChange": 0.5,
            "GapPct": 0.1, "BreakoutPos20D": 0.95, "IsBreakout": False,
            "PreBreakoutProbRaw": raw, "PreBreakoutProb": raw, "PreBreakoutProb%": round(raw * 100, 1)}


@unittest.skipUnless(_PANDAS, "pandas not installed")
class EngineSinkTests(unittest.TestCase):
    def test_sink_takes_near_setups_and_production_keeps_top_n(self):
        import pandas as pd

        from scan import engine

        rows = [{"Ticker": f"S{i}", "BreakoutScore": 50 - i, "IsBreakout": i < 2, "BreakoutPos20D": 0.99} for i in range(5)]
        rows += [{"Ticker": f"N{i}", "BreakoutScore": 7 - i, "IsBreakout": False, "BreakoutPos20D": 0.95} for i in range(4)]
        rows += [{"Ticker": "HIGH", "BreakoutScore": 3, "IsBreakout": False, "BreakoutPos20D": 1.0}]
        full = pd.DataFrame(rows)
        seen = {}

        def fake_scan(**kw):
            seen["top_n"] = kw["top_n"]
            return full.head(kw["top_n"])

        sink = {"n": 3}
        bars = pd.DataFrame({"Open": [10.0] * 25, "High": [10.5] * 25, "Low": [9.5] * 25,
                             "Close": [10.0] * 25, "Volume": [1000] * 25},
                            index=pd.date_range("2026-01-01", periods=25, freq="D"))
        with mock.patch("scan.engine.st.session_state", {"show_scan_progress": False}), \
                mock.patch("data.prices.fetch_price_data_parallel", return_value=({"AAA": bars, "SPY": bars}, [])), \
                mock.patch("data.prices.fetch_price_data_batch", side_effect=AssertionError("no batch")), \
                mock.patch("scan.breakout.run_breakout_scan", side_effect=fake_scan):
            out = engine.run_breakout_scan(["AAA"], premarket=False, afterhours=False, unusual_volume=False,
                                           min_gap=0.0, min_price=1.0, max_price=100.0, top_n=4,
                                           diagnostics=False, use_cache=False, prebreakout_sink=sink)
        self.assertGreater(seen["top_n"], 4)
        self.assertEqual(list(out["Ticker"]), ["S0", "S1", "S2", "S3"])
        self.assertEqual([r["Ticker"] for r in sink["rows"]], ["N0", "N1", "N2"])
        self.assertEqual(sink["eligible"], 4)  # HIGH sits at its 20-day high: not a candidate


@unittest.skipUnless(_PANDAS, "pandas not installed")
class CronSaveTests(unittest.TestCase):
    def test_scores_and_saves_candidates_model_order(self):
        import datetime as dt

        from scheduler import cron_runner

        def fake_score(df):
            df = df.copy()
            df["PreBreakoutProbRaw"] = [0.1, 0.3, 0.2]
            df["PreBreakoutProb"] = df["PreBreakoutProbRaw"]
            df["PreBreakoutProb%"] = df["PreBreakoutProbRaw"] * 100
            return df

        rows = [{"Ticker": t, "BreakoutScore": 5.0, "Last": 10.0} for t in ("AAA", "BBB", "CCC")]
        started = dt.datetime(2026, 11, 13, 14, 35, tzinfo=dt.timezone.utc)
        with mock.patch("ml_prebreakout.load_prebreakout_model", return_value={"model_version": "v9"}), \
                mock.patch("ml_prebreakout.score_prebreakout", side_effect=fake_score), \
                mock.patch("db.prebreakout_candidates.save_candidates", return_value=True) as save:
            cron_runner._save_prebreakout_candidates({"rows": rows, "eligible": 3}, universe="US_MARKET",
                                                     scan_started_at=started)
        args, kwargs = save.call_args
        self.assertEqual(args[0], "US_MARKET")
        self.assertEqual([r["Ticker"] for r in args[3]], ["BBB", "CCC", "AAA"])
        self.assertEqual(kwargs["model_version"], "v9")

    def test_failures_never_raise(self):
        import datetime as dt

        from scheduler import cron_runner

        with mock.patch("ml_prebreakout.load_prebreakout_model", side_effect=RuntimeError("boom")):
            cron_runner._save_prebreakout_candidates({"rows": [{"Ticker": "A"}]}, universe="US_MARKET",
                                                     scan_started_at=dt.datetime.now(dt.timezone.utc))
        cron_runner._save_prebreakout_candidates({"rows": []}, universe="US_MARKET",
                                                 scan_started_at=dt.datetime.now(dt.timezone.utc))


@unittest.skipUnless(_PANDAS, "pandas not installed")
class TabTests(unittest.TestCase):
    def _latest(self, latest):
        from api import scans

        scans._cached.__globals__["_cache"].clear() if hasattr(scans._cached.__globals__.get("_cache"), "clear") else None
        opps = [{"ticker": chr(65 + i), "score": 90 - i, "signals": [], "breakout_score": 20.0,
                 "prob_raw": i / 100, "prob": 20.0 + i, "prob_rank": 12 - i} for i in range(12)]
        import datetime as dt
        run = {"id": 991, "created_at": dt.datetime(2026, 11, 13, 15, 0, tzinfo=dt.timezone.utc)}
        with mock.patch.object(scans, "market_runs", return_value=[run]), \
                mock.patch.object(scans, "run_opportunities", return_value=opps), \
                mock.patch("db.prebreakout_candidates.load_latest", return_value=latest), \
                mock.patch.object(scans, "scan_freshness", return_value={"stale": False}), \
                mock.patch.object(scans, "run_freshness", return_value={}), \
                mock.patch.object(scans, "_iso", return_value=None):
            all_rows = scans.latest_scan({"can_early_breakout": True}, "premium", limit=50, offset=0)
            tab = scans.latest_scan({"can_early_breakout": True}, "premium", limit=50, offset=0, signal="prebreakout")
            free = scans.latest_scan({"can_early_breakout": False}, "basic", limit=50, offset=0, signal="prebreakout")
        return all_rows, tab, free

    def test_tab_lists_top_candidates_and_setups_hide_percent(self):
        import datetime as dt

        rows = [_cand(f"C{i:02d}", raw=i / 100) for i in range(15)]
        latest = {"scan_id": "x", "scanned_at": dt.datetime(2026, 11, 13, 14, 35, tzinfo=dt.timezone.utc),
                  "model_version": "v9", "rows": rows}
        all_rows, tab, free = self._latest(latest)
        self.assertTrue(all(s["prob"] is None for s in all_rows["setups"]))  # setups: % hidden
        self.assertEqual([s["ticker"] for s in tab["setups"]], [f"C{i:02d}" for i in range(14, 4, -1)])
        self.assertTrue(all(s["rank"] is None and s["prob"] is not None for s in tab["setups"]))
        self.assertEqual(free["setups"], [])

    def test_stale_candidates_fall_back_to_scan_rows(self):
        import datetime as dt

        latest = {"scan_id": "x", "scanned_at": dt.datetime(2026, 11, 10, 14, 35, tzinfo=dt.timezone.utc),
                  "model_version": "v9", "rows": [_cand("OLD", 0.5)]}
        _all, tab, _free = self._latest(latest)
        self.assertNotIn("OLD", [s["ticker"] for s in tab["setups"]])


@unittest.skipUnless(_PANDAS, "pandas not installed")
class AlertTests(unittest.TestCase):
    def test_prebreakout_alert_uses_candidates_and_filters(self):
        from api.alert_rules import scan_matches
        from ui.results_intelligence import consolidate_scanner_results

        cands = consolidate_scanner_results([_cand("AAA", 0.3, last=5.0), _cand("BBB", 0.2, last=50.0)], top_n=None)
        cands.sort(key=lambda o: -o["prob_raw"])
        rule = {"value": json.dumps({"setup": "prebreakout", "min_price": 10}), "threshold": 0}
        with mock.patch("api.today.market_runs", return_value=[{"id": 1}]), \
                mock.patch("api.scans.prebreakout_candidates", return_value=cands):
            self.assertEqual(scan_matches({"ranked": {}}, rule, premium=True), ["BBB"])
            self.assertEqual(scan_matches({"ranked": {}}, rule, premium=False), [])


if __name__ == "__main__":
    unittest.main()
