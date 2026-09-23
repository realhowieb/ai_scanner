"""Run 44 — canonical US_MARKET universe tests (mocked provider, no network)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import us_market_universe as um


def _assets(n=1500, prefix="AA"):
    """n plausible active tradable us_equity assets on NASDAQ."""
    out = []
    for i in range(n):
        out.append({"symbol": f"{prefix}{i}", "status": "active", "tradable": True,
                    "class": "us_equity", "exchange": "NASDAQ"})
    return out


class FilterTests(unittest.TestCase):
    def test_active_tradable_kept_and_sorted_deduped(self):
        assets = _assets(3) + _assets(3)  # duplicates
        res = um.filter_assets(assets)
        self.assertEqual(res["symbols"], sorted(set(res["symbols"])))
        self.assertEqual(res["exclusions"]["duplicate"], 3)

    def test_inactive_non_tradable_excluded(self):
        assets = [
            {"symbol": "AAA", "status": "active", "tradable": True, "class": "us_equity", "exchange": "NASDAQ"},
            {"symbol": "DEAD", "status": "inactive", "tradable": True, "class": "us_equity", "exchange": "NASDAQ"},
            {"symbol": "NOTR", "status": "active", "tradable": False, "class": "us_equity", "exchange": "NASDAQ"},
        ]
        res = um.filter_assets(assets)
        self.assertEqual(res["symbols"], ["AAA"])
        self.assertEqual(res["exclusions"]["inactive"], 1)
        self.assertEqual(res["exclusions"]["non_tradable"], 1)

    def test_malformed_and_unsupported_rejected(self):
        assets = [
            {"symbol": "AAA", "status": "active", "tradable": True, "class": "us_equity", "exchange": "NYSE"},
            {"symbol": "", "status": "active", "tradable": True, "class": "us_equity", "exchange": "NYSE"},
            {"symbol": "TOOLONGSYMBOL", "status": "active", "tradable": True, "class": "us_equity", "exchange": "NYSE"},
            {"symbol": "ABC.WS", "status": "active", "tradable": True, "class": "us_equity", "exchange": "NYSE"},  # warrant
            {"symbol": "XYZ", "status": "active", "tradable": True, "class": "crypto", "exchange": "FTXU"},
            {"symbol": "FOREIGN", "status": "active", "tradable": True, "class": "us_equity", "exchange": "TSX"},
        ]
        res = um.filter_assets(assets)
        self.assertEqual(res["symbols"], ["AAA"])
        self.assertGreaterEqual(res["exclusions"]["malformed_symbol"], 1)
        self.assertGreaterEqual(res["exclusions"]["unsupported_asset_type"], 1)
        self.assertGreaterEqual(res["exclusions"]["wrong_exchange"], 1)

    def test_preferred_shares_excluded_commons_kept(self):
        # Preferred-class symbols (.PR<letter>) are illiquid, cause provider
        # timeouts / no-data, and are not scan targets → excluded. Common stocks
        # that merely start with PR/PS must NOT be excluded.
        def a(sym):
            return {"symbol": sym, "status": "active", "tradable": True,
                    "class": "us_equity", "exchange": "NYSE"}
        prefs = ["PSA.PRF", "PRIF.PRD", "PSEC.PRA", "PSA.PRK"]
        commons = ["PRE", "PRI", "PRG", "PSA", "PRGO", "PSX"]
        res = um.filter_assets([a(s) for s in prefs + commons])
        kept = set(res["symbols"])
        self.assertEqual(res["exclusions"]["preferred_share"], len(prefs))
        for c in commons:
            self.assertIn(um.normalize_ticker(c), kept)  # common stocks preserved
        self.assertFalse(any(um._is_preferred(um.normalize_ticker(c)) for c in commons))

    def test_deterministic(self):
        self.assertEqual(um.filter_assets(_assets(50)), um.filter_assets(_assets(50)))


class BuildTests(unittest.TestCase):
    def _tmp_cache(self):
        d = tempfile.mkdtemp()
        return Path(d) / "us_market.json"

    def test_live_source_and_no_2000_cap(self):
        with patch.object(um, "CACHE_PATH", self._tmp_cache()):
            res = um.build_us_market_universe(fetch=lambda: _assets(3000))
        self.assertEqual(res["source"], "live")
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["symbol_count"], 3000)  # NOT capped at 2000

    def test_cached_fallback_on_provider_failure(self):
        cache = self._tmp_cache()
        with patch.object(um, "CACHE_PATH", cache):
            um.build_us_market_universe(fetch=lambda: _assets(1500))  # seeds cache
            res = um.build_us_market_universe(fetch=lambda: None)     # provider down
        self.assertEqual(res["source"], "cached")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["symbol_count"], 1500)

    def test_cached_cannot_masquerade_as_live(self):
        cache = self._tmp_cache()
        with patch.object(um, "CACHE_PATH", cache):
            um.build_us_market_universe(fetch=lambda: _assets(1500))
            res = um.build_us_market_universe(fetch=lambda: None)
        self.assertNotEqual(res["source"], "live")
        self.assertTrue(res["is_fallback"])
        self.assertIn("fallback_reason", res)

    def test_no_live_no_cache_is_explicit_failure(self):
        with patch.object(um, "CACHE_PATH", self._tmp_cache()):
            res = um.build_us_market_universe(fetch=lambda: None)
        self.assertEqual(res["source"], "none")
        self.assertEqual(res["symbols"], [])
        self.assertTrue(res["is_fallback"])

    def test_implausibly_small_live_not_trusted(self):
        with patch.object(um, "CACHE_PATH", self._tmp_cache()):
            res = um.build_us_market_universe(fetch=lambda: _assets(10),
                                              allow_cache_fallback=False)
        self.assertEqual(res["source"], "none")  # 10 < _MIN_PLAUSIBLE


class SchedulerIntegrationTests(unittest.TestCase):
    def test_default_scheduled_universe_is_us_market(self):
        from scheduler import cron_runner
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CRON_UNIVERSES", None)
            self.assertEqual(cron_runner._configured_universes(), ["US_MARKET"])

    def test_explicit_cron_universes_overrides(self):
        import os

        from scheduler import cron_runner
        with patch.dict(os.environ, {"CRON_UNIVERSES": "SP500,COMBO"}):
            self.assertEqual(cron_runner._configured_universes(), ["SP500", "COMBO"])

    def test_us_market_loader_uses_provider_no_cap(self):
        from scheduler import cron_runner
        with patch("data.us_market_universe.build_us_market_universe",
                   return_value={"source": "live", "symbol_count": 5000,
                                 "symbols": [f"S{i}" for i in range(5000)],
                                 "is_fallback": False, "generated_at": "t"}):
            syms, meta = cron_runner._load_universe_result("US_MARKET")
        self.assertEqual(len(syms), 5000)      # no 2,000 cap
        self.assertEqual(meta["universe_source"], "live")

    def test_legacy_universes_backward_compatible(self):
        from scheduler import cron_runner
        with patch.object(cron_runner, "_read_symbols",
                          side_effect=lambda p: ["AAA", "BBB"] if "sp500" in str(p).lower()
                          else ["CCC", "DDD", "EEE"]):
            sp500, m1 = cron_runner._load_universe_result("SP500")
            combo, m2 = cron_runner._load_universe_result("COMBO")
        self.assertEqual(sp500, ["AAA", "BBB"])
        self.assertEqual(m1["universe_source"], "static")
        self.assertIn("AAA", combo)
        self.assertIn("CCC", combo)


class CoverageIntegrityTests(unittest.TestCase):
    def test_low_priced_not_healthy(self):
        from analytics.coverage import build_coverage_funnel, classify_health
        # 7,000 eligible, only 1,500 priced → must NOT be HEALTHY.
        funnel = build_coverage_funnel(universe_version="US_MARKET", expected=7000,
                                       eligible=7000, attempted=7000, price_success=1500,
                                       skipped=[("X", "timeout")] * 5500)
        health = classify_health(funnel)
        self.assertNotEqual(health["state"], "HEALTHY")
        self.assertLess(funnel["coverage_pct"], 0.5)

    def test_top_n_does_not_limit_universe(self):
        # The distinction: universe size (eligible/attempted) is independent of the
        # returned candidate count (results). A 7000-symbol scan returning 100
        # candidates is full coverage, not a 100-symbol scan.
        from analytics.coverage import build_coverage_funnel
        funnel = build_coverage_funnel(universe_version="US_MARKET", expected=7000,
                                       eligible=7000, attempted=7000, price_success=6900,
                                       skipped=[], results=100)
        self.assertEqual(funnel["counts"]["attempted"], 7000)   # evaluated ~all
        self.assertEqual(funnel["counts"]["results"], 100)      # only display cap
        self.assertGreater(funnel["coverage_pct"], 0.95)


class BatchSizeTests(unittest.TestCase):
    def test_cron_batch_size_env_used_and_clamped(self):
        from config import (
            PRICE_FETCH_CHUNK_MAX,
            PRICE_FETCH_CHUNK_MIN,
            PRICE_FETCH_CHUNK_SIZE,
        )
        from scan.engine import resolve_chunk_size
        # session value wins over env
        self.assertEqual(resolve_chunk_size(120, "50"),
                         max(PRICE_FETCH_CHUNK_MIN, min(PRICE_FETCH_CHUNK_MAX, 120)))
        # env used when no session value (scheduled/headless path)
        self.assertEqual(resolve_chunk_size(None, "40"),
                         max(PRICE_FETCH_CHUNK_MIN, min(PRICE_FETCH_CHUNK_MAX, 40)))
        # default when neither set
        self.assertEqual(resolve_chunk_size(None, None),
                         max(PRICE_FETCH_CHUNK_MIN, min(PRICE_FETCH_CHUNK_MAX, PRICE_FETCH_CHUNK_SIZE)))
        # clamped + invalid safe
        self.assertEqual(resolve_chunk_size(None, "999999"), PRICE_FETCH_CHUNK_MAX)
        self.assertEqual(resolve_chunk_size(None, "garbage"),
                         max(PRICE_FETCH_CHUNK_MIN, min(PRICE_FETCH_CHUNK_MAX, PRICE_FETCH_CHUNK_SIZE)))


if __name__ == "__main__":
    unittest.main()
