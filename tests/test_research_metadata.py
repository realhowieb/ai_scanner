"""Run 57 — point-in-time research metadata (persist what the scanner knew)."""
import copy
import datetime as dt
import json
import sqlite3
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from analytics import forward_readiness as fr
from analytics import research_metadata as rm
from analytics.observation_capture import build_scan_observations, capture_scan_observations
from analytics.research_cohorts import (
    build_control_observations,
    build_near_miss_observations,
    select_control_symbols,
)

TS = "2026-09-28T13:35:00+00:00"
KW = dict(premarket=False, afterhours=False, unusual_volume=False, min_gap=0.0, min_price=1.0,
          max_price=1000.0, top_n=10, min_dollar_vol=5e6, profile="regular", use_cache=False)


def _frames(n=40, days=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-07-01", periods=days, freq="B")
    out = {}
    for i in range(n + 1):
        sym = "SPY" if i == n else f"T{i:02d}"
        c = 20 * np.cumprod(1 + rng.normal(0.003, 0.02, days))
        df = pd.DataFrame({"Open": c * 0.99, "High": c * 1.02, "Low": c * 0.98, "Close": c,
                           "Adj Close": c, "Volume": rng.integers(2e6, 5e6, days).astype(float)},
                          index=idx)
        df.attrs["source"] = "alpaca_multi" if i % 5 else "rescue_single"
        df.attrs["feed"] = "iex" if i % 5 else None
        out[sym] = df
    return out


_DATA = _frames()


def _scan(research_sink=None):
    """Frozen scanner fixture: the real engine on fixed synthetic prices."""
    from scan import engine

    def fake(tickers, use_cache=True):
        return ({k: _DATA[k].copy() for k in tickers if k in _DATA}, [])
    with mock.patch("data.prices.fetch_price_data_parallel", side_effect=fake):
        return engine.run_breakout_scan([f"T{i:02d}" for i in range(40)],
                                        research_sink=research_sink, **KW)


def _ctx(**over):
    with mock.patch.dict("os.environ", {"GITHUB_SHA": "a" * 40}):
        return rm.build_run_context(universe="US_MARKET", session="regular", scan_id=TS,
                                    scan_config={"top_n": 10, "near_miss_n": 5, "afterhours": False},
                                    price_meta=over.get("price_meta"))


def _strip(obs):
    out = []
    for o in obs:
        o = copy.deepcopy(o)
        o.pop("research_metadata", None)
        out.append(o)
    return out


class FrozenScannerTests(unittest.TestCase):
    """15-18: ranking, scores and cohort membership are unchanged by Run 57."""

    @classmethod
    def setUpClass(cls):
        cls.plain = _scan()
        cls.sink = {"near_miss_n": 5}
        cls.with_sink = _scan(cls.sink)

    def test_ranking_and_scores_unchanged(self):
        self.assertTrue(self.plain.equals(self.with_sink))
        self.assertEqual(list(self.plain.Ticker), list(self.with_sink.Ticker))

    def test_provider_tags_are_read_not_fetched(self):
        snap = self.sink["price_snapshot"]
        self.assertEqual(snap["T01"]["source"], "alpaca_multi")
        self.assertEqual((snap["T00"]["source"], snap["T00"]["feed"]), ("rescue_single", None))
        self.assertEqual(set(snap["T01"]), {"price", "volume", "source", "feed"})

    def test_cohort_membership_unchanged_by_metadata(self):
        rows = self.with_sink.to_dict("records")
        ctx = _ctx(price_meta=self.sink["price_snapshot"])
        plain_c = build_scan_observations(rows, universe="US_MARKET", scan_timestamp=TS, scan_id=TS,
                                          research_cohort="CANDIDATE")
        meta_c = copy.deepcopy(plain_c)
        rm.attach(meta_c, ctx, rows=rows, rank_offset=0, price_meta=self.sink["price_snapshot"])
        nm_plain = build_near_miss_observations(self.sink["near_miss_rows"], universe="US_MARKET",
                                                scan_timestamp=TS, scan_id=TS)
        nm_meta = build_near_miss_observations(self.sink["near_miss_rows"], universe="US_MARKET",
                                               scan_timestamp=TS, scan_id=TS, research_run_context=ctx,
                                               top_n=10, price_meta=self.sink["price_snapshot"])
        syms = select_control_symbols(self.sink["evaluated_symbols"], scan_run_id=TS,
                                      exclude=[r["Ticker"] for r in rows], n=8)
        c_plain = build_control_observations(syms, self.sink["price_snapshot"], universe="US_MARKET",
                                             scan_timestamp=TS, scan_id=TS)
        c_meta = build_control_observations(syms, self.sink["price_snapshot"], universe="US_MARKET",
                                            scan_timestamp=TS, scan_id=TS, research_run_context=ctx)
        for plain, meta in ((plain_c, meta_c), (nm_plain, nm_meta), (c_plain, c_meta)):
            self.assertEqual([o["observation_id"] for o in plain], [o["observation_id"] for o in meta])
            self.assertEqual(json.dumps(_strip(meta), sort_keys=True, default=str),
                             json.dumps(plain, sort_keys=True, default=str))
            self.assertTrue(all("research_metadata" in o for o in meta))

    def test_candidate_and_near_miss_ranks_follow_scanner_order(self):
        rows = self.with_sink.to_dict("records")
        ctx = _ctx()
        cand = capture_scan_observations(rows, universe="US_MARKET", scan_timestamp=TS, scan_id=TS,
                                         dry_run=True, research_cohort="CANDIDATE")
        self.assertEqual(cand["would_write"], 10)
        obs = build_scan_observations(rows, universe="US_MARKET", scan_timestamp=TS, scan_id=TS)
        rm.attach(obs, ctx, rows=rows, rank_offset=0)
        self.assertEqual([o["research_metadata"]["rank_at_observation"] for o in obs], list(range(1, 11)))
        nm = build_near_miss_observations(self.sink["near_miss_rows"], universe="US_MARKET",
                                          scan_timestamp=TS, scan_id=TS, research_run_context=ctx, top_n=10)
        self.assertEqual([o["research_metadata"]["rank_at_observation"] for o in nm], list(range(11, 16)))
        feats = obs[0]["research_metadata"]["row_features"]
        self.assertAlmostEqual(feats["breakout_pos_20d"], float(rows[0]["BreakoutPos20D"]))
        self.assertIn(feats.get("ema_cross"), ("Golden", "Death", None))


class TierRegimeTests(unittest.TestCase):
    def test_tier_is_null_not_recomputed_for_every_cohort(self):
        ctx = _ctx()
        rows = [{"Ticker": "AAA", "BreakoutScore": 90.0, "Last": 10, "VolRel20": 5, "IsBreakout": True}]
        cand = build_scan_observations(rows, universe="US_MARKET", scan_timestamp=TS, scan_id=TS)
        rm.attach(cand, ctx, rows=rows)
        ctrl = build_control_observations(["BBB"], {"BBB": {"price": 5.0, "volume": 1e5}},
                                          universe="US_MARKET", scan_timestamp=TS, research_run_context=ctx)
        for o in cand + ctrl:
            m = o["research_metadata"]
            self.assertIsNone(m["tier_at_observation"])  # even a 90-score breakout gets no tier
            self.assertEqual(m["tier_source"], rm.TIER_UNAVAILABLE)
            self.assertIsNone(m["tier_version"])

    def test_regime_is_null_and_never_reconstructed(self):
        m = _ctx()
        self.assertIsNone(m["market_regime_at_observation"])
        self.assertEqual(m["market_regime_source"], rm.REGIME_UNAVAILABLE)
        with mock.patch("ui.opportunities.classify_market_regime") as clf:
            _ctx()
            clf.assert_not_called()

    def test_existing_regime_passthrough_still_persisted_exactly(self):
        o = build_scan_observations([{"Ticker": "AAA", "Last": 10}], universe="US_MARKET",
                                    scan_timestamp=TS, market_regime="RISK-ON")[0]
        self.assertEqual(o["market_context"]["market_regime"], "RISK-ON")


class ProvenanceTests(unittest.TestCase):
    def test_scoring_provenance_persisted(self):
        m = rm.observation_metadata(_ctx(), symbol="AAA", rank=3,
                                    price_meta={"AAA": {"source": "alpaca_multi", "feed": "sip"}})
        self.assertEqual(m["scoring_version"], "breakout-1")
        self.assertEqual(m["scanner_commit_sha"], "a" * 40)
        self.assertEqual((m["universe_name"], m["scan_mode"]), ("US_MARKET", "scheduled"))
        self.assertEqual((m["price_provider"], m["price_feed"]), ("alpaca_multi", "sip"))
        self.assertEqual(m["scan_config"]["top_n"], 10)
        self.assertEqual(m["schema_version"], rm.RESEARCH_METADATA_SCHEMA)

    def test_commit_sha_optional_and_validated(self):
        with mock.patch.dict("os.environ", {"GITHUB_SHA": "", "HSF_COMMIT_SHA": ""}):
            self.assertIsNone(rm.commit_sha())
        with mock.patch.dict("os.environ", {"GITHUB_SHA": "not a sha"}):
            self.assertIsNone(rm.commit_sha())

    def test_run_level_provider_mix(self):
        ctx = rm.build_run_context(universe="U", session=None, scan_id=None,
                                   price_meta={"A": {"source": "alpaca_multi", "feed": "iex"},
                                               "B": {"source": None}})
        self.assertEqual(ctx["price_sources"], {"alpaca_multi": 1, "untagged": 1})


class PointInTimeTests(unittest.TestCase):
    def test_no_outcome_fields_can_leak(self):
        for key in ("raw_return", "directional_return", "mfe", "mae", "future_high",
                    "evaluation_time", "matured", "horizon", "outcomes", "hindsight_regime"):
            with self.assertRaises(ValueError, msg=key):
                rm.assert_point_in_time({"row_features": {key: 1}})
        rm.assert_point_in_time(_ctx())

    def test_row_features_only_from_ranked_row_columns(self):
        row = {"Ticker": "AAA", "BreakoutPos20D": 0.99, "Trend20D%": float("nan"),
               "Spark10D": [1, 2], "future_return": 0.5}
        m = rm.observation_metadata(_ctx(), symbol="AAA", row=row)
        self.assertEqual(m["row_features"], {"breakout_pos_20d": 0.99})

    def test_attach_is_non_fatal(self):
        obs = [{"symbol": "AAA"}]
        with mock.patch.object(rm, "observation_metadata", side_effect=RuntimeError("boom")):
            out = rm.attach(obs, _ctx(), rows=[{"Ticker": "AAA"}])
        self.assertEqual(out, [{"symbol": "AAA"}])

    def test_capture_non_fatal_when_context_missing(self):
        stats = capture_scan_observations([{"Ticker": "AAA", "Last": 1}], universe="U",
                                          scan_timestamp=TS, dry_run=True, research_run_context=None)
        self.assertEqual(stats["would_write"], 1)


class SchemaCompatibilityTests(unittest.TestCase):
    def test_legacy_rows_readable_and_not_backfilled(self):
        from db import hsf_observations as store
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            legacy = build_scan_observations([{"Ticker": "OLD", "Last": 1}], universe="U",
                                             scan_timestamp="2026-09-20T13:35:00+00:00")[0]
            new = build_scan_observations([{"Ticker": "NEW", "Last": 1}], universe="U", scan_timestamp=TS)
            rm.attach(new, _ctx(), rows=[{"Ticker": "NEW"}])
            self.assertTrue(store.save_observation(legacy, conn=conn))
            self.assertTrue(store.save_observation(new[0], conn=conn))
            got = {o["symbol"]: o for o in store.load_recent_observations(limit=10, conn=conn)}
            self.assertNotIn("research_metadata", got["OLD"])  # NULL, never synthesized
            self.assertEqual(got["NEW"]["research_metadata"]["scoring_version"], "breakout-1")
            # first-write-wins: re-saving the legacy id with metadata does not rewrite it
            again = copy.deepcopy(legacy)
            again["research_metadata"] = {"tier_at_observation": "STRONG"}
            self.assertFalse(store.save_observation(again, conn=conn))
            self.assertNotIn("research_metadata",
                             store.load_observation(legacy["observation_id"], conn=conn))
        finally:
            conn.close()

    def test_metadata_block_does_not_change_ids_or_completeness(self):
        a = build_scan_observations([{"Ticker": "AAA", "Last": 1, "VolRel20": 2}], universe="U", scan_timestamp=TS)
        b = copy.deepcopy(a)
        rm.attach(b, _ctx(), rows=[{"Ticker": "AAA"}])
        self.assertEqual(a[0]["observation_id"], b[0]["observation_id"])
        self.assertEqual(a[0]["data_quality"], b[0]["data_quality"])


class Run56CompatibilityTests(unittest.TestCase):
    def test_monitor_reports_completeness_and_stays_clean(self):
        from tests.test_forward_readiness import dataset, now_after
        obs, outs = dataset(3)
        for o in obs:
            rm.attach([o], _ctx(), rows=[{"Ticker": o["symbol"]}])
        r = fr.monitor(obs, outs, now=now_after(3))
        mc = r["metadata_completeness"]
        self.assertEqual(mc["CANDIDATE"]["metadata_block_coverage_pct"], 100.0)
        self.assertEqual(mc["CONTROL"]["tier_metadata_coverage_pct"], 0.0)
        self.assertEqual(mc["CONTROL"]["scoring_version_coverage_pct"], 100.0)
        self.assertEqual(fr.forbidden_keys(r), [])
        self.assertEqual(r["epoch"]["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")

    def test_metadata_audit_is_observation_only(self):
        from scripts import audit_research_metadata as audit
        from tests.test_forward_readiness import dataset
        obs, _ = dataset(2)
        rm.attach(obs[:5], _ctx(), rows=[{"Ticker": o["symbol"]} for o in obs[:5]])
        r = audit.audit(obs)
        self.assertEqual(fr.forbidden_keys(r), [])
        self.assertEqual(r["coverage_by_cohort"]["CANDIDATE"]["with_metadata_block"], 5)
        text = json.dumps(r).lower()
        for bad in ("win_rate", "mean_return", "spearman", "raw_return", "directional_return"):
            self.assertNotIn(bad, text)
        self.assertIn("## Coverage by cohort", audit.render_markdown(r))

    def test_epoch_not_reset(self):
        self.assertEqual(fr.FORWARD_EPOCH["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")


class CronWiringTests(unittest.TestCase):
    def test_research_context_non_fatal(self):
        from scheduler import cron_runner
        with mock.patch("analytics.research_metadata.build_run_context", side_effect=RuntimeError("x")):
            self.assertIsNone(cron_runner._research_run_context(
                universe="U", scan_id=TS, scan_params={"top_n": 10}, research_sink={}))
        ctx = cron_runner._research_run_context(universe="U", scan_id=TS,
                                                scan_params={"top_n": 10, "afterhours": False},
                                                research_sink={"near_miss_n": 5, "price_snapshot": {}})
        self.assertEqual((ctx["scan_config"]["top_n"], ctx["scan_config"]["near_miss_n"]), (10, 5))


if __name__ == "__main__":
    unittest.main()
