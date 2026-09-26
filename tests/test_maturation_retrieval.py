"""Run 51 (maturation hardening) — batched/cached/retrying minute-bar retrieval."""
import datetime as dt
import sqlite3
import unittest
from unittest import mock

from analytics import observation_capture as oc
from data import price_alpaca as pa
from db import hsf_observations as store
from scripts import mature_observations as worker

NOW = "2026-08-25T18:00:00+00:00"
CFG = {"data_url": "https://x", "api_key": "k", "api_secret": "s"}


def _obs(ticker="NVDA", scan_ts="2026-08-25T14:00:00+00:00"):
    return oc.build_scan_observations(
        [{"Ticker": ticker, "BreakoutScore": 8.0, "Last": 100.0, "Volume": 1e6,
          "PctChange": 1.0, "GapPct": 0.5, "Trend20D%": 2.0, "Trend10D%": 1.0,
          "VolRel20": 2.5, "DollarVol20": 5e8, "Volatility20D%": 3.0, "IsBreakout": True}],
        universe="US_MARKET", scan_timestamp=scan_ts, session="morning")[0]


def _bars(n=200, base=100.0):
    t0 = dt.datetime(2026, 8, 25, 14, 0, tzinfo=dt.timezone.utc)
    return [{"t": (t0 + dt.timedelta(minutes=i)).isoformat(), "o": base, "h": base,
             "l": base, "c": base + i * 0.1, "v": 1000} for i in range(n)]


def _resp(status=200, payload=None, headers=None):
    r = mock.MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = payload if payload is not None else {}
    if status >= 400:
        r.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


def _patched(responses):
    fake = mock.MagicMock()
    fake.get.side_effect = list(responses)
    return (mock.patch.object(pa, "requests", fake),
            mock.patch.object(pa, "get_alpaca_config", return_value=CFG),
            mock.patch.object(pa, "get_alpaca_data_feed", return_value="iex"),
            fake)


class RetryingGetTests(unittest.TestCase):
    def test_429_respects_retry_after_then_succeeds(self):
        p1, p2, p3, fake = _patched([_resp(429, headers={"Retry-After": "2"}),
                                     _resp(200, {"ok": 1})])
        sleeps, stats = [], pa.AlpacaRequestStats()
        with p1, p2, p3:
            out = pa._alpaca_get("u", headers={}, params={}, timeout_s=1,
                                 stats=stats, sleep=sleeps.append)
        self.assertEqual(out, {"ok": 1})
        self.assertEqual((stats.requests, stats.rate_limited, stats.retries), (2, 1, 1))
        self.assertGreaterEqual(sleeps[0], 2.0)
        self.assertLess(sleeps[0], 2.3)

    def test_persistent_429_raises_rate_limit_error_bounded(self):
        p1, p2, p3, fake = _patched([_resp(429)] * 4)
        stats = pa.AlpacaRequestStats()
        with p1, p2, p3, self.assertRaises(pa.AlpacaRateLimitError) as cm:
            pa._alpaca_get("u", headers={}, params={}, timeout_s=1, stats=stats,
                           max_retries=3, sleep=lambda s: None)
        self.assertTrue(getattr(cm.exception, "rate_limited", False))
        self.assertEqual((stats.requests, stats.rate_limited, stats.retries), (4, 4, 3))

    def test_backoff_is_exponential_with_jitter_and_capped(self):
        for attempt in range(8):
            d = pa._backoff_delay(attempt, 1.0, 30.0)
            cap = min(30.0, 2 ** attempt)
            self.assertGreaterEqual(d, cap / 2)
            self.assertLessEqual(d, cap)

    def test_5xx_retried(self):
        p1, p2, p3, fake = _patched([_resp(503), _resp(200, {"ok": 1})])
        stats = pa.AlpacaRequestStats()
        with p1, p2, p3:
            self.assertEqual(pa._alpaca_get("u", headers={}, params={}, timeout_s=1,
                                            stats=stats, sleep=lambda s: None), {"ok": 1})
        self.assertEqual((stats.rate_limited, stats.retries), (0, 1))

    def test_client_error_not_retried(self):
        p1, p2, p3, fake = _patched([_resp(403)])
        with p1, p2, p3, self.assertRaises(RuntimeError):
            pa._alpaca_get("u", headers={}, params={}, timeout_s=1, sleep=lambda s: None)
        self.assertEqual(fake.get.call_count, 1)

    def test_single_symbol_fetch_retries_429(self):
        page = {"bars": [{"t": "T1", "c": 1.0}], "next_page_token": None}
        p1, p2, p3, fake = _patched([_resp(429), _resp(200, page)])
        with p1, p2, p3, mock.patch.object(pa.time, "sleep"):
            self.assertEqual([b["t"] for b in pa.fetch_minute_bars("NVDA", "2026-09-01")], ["T1"])


class MultiSymbolFetchTests(unittest.TestCase):
    def test_paginates_maps_class_shares_one_series(self):
        pages = [
            _resp(200, {"bars": {"AAPL": [{"t": "T1", "c": 1}], "BRK.B": [{"t": "T1", "c": 2}]},
                        "next_page_token": "p2"}),
            _resp(200, {"bars": {"BRK.B": [{"t": "T2", "c": 3}]}, "next_page_token": None}),
        ]
        p1, p2, p3, fake = _patched(pages)
        stats = pa.AlpacaRequestStats()
        with p1, p2, p3:
            out = pa.fetch_minute_bars_multi(["AAPL", "BRK-B", "MSFT"], "S", "E",
                                             stats=stats, sleep=lambda s: None)
        self.assertEqual(stats.requests, 2)
        self.assertEqual([b["t"] for b in out["BRK-B"]], ["T1", "T2"])
        self.assertNotIn("MSFT", out)  # no bars → absent (true missing data)
        params = fake.get.call_args_list[0].kwargs["params"]
        self.assertEqual(params["symbols"], "AAPL,BRK.B,MSFT")
        self.assertEqual((params["start"], params["end"]), ("S", "E"))

    def test_mid_series_failure_never_returns_partial(self):
        pages = [_resp(200, {"bars": {"AAPL": [{"t": "T1", "c": 1}]}, "next_page_token": "p2"}),
                 _resp(403)]
        p1, p2, p3, fake = _patched(pages)
        with p1, p2, p3, self.assertRaises(RuntimeError):
            pa.fetch_minute_bars_multi(["AAPL"], "S", sleep=lambda s: None)


class BatchedWorkerTests(unittest.TestCase):
    def _obs_set(self, n_symbols, per_symbol=3):
        out = []
        for i in range(n_symbols):
            for h in range(per_symbol):
                out.append(_obs(f"S{i:03d}", f"2026-08-25T{14 + h:02d}:00:00+00:00"))
        return out

    def test_batches_symbols_and_reuses_bars_per_symbol(self):
        calls = []

        def batch(symbols, start, end):
            calls.append((list(symbols), start, end))
            return {s: _bars() for s in symbols}
        r = worker.mature_observations(self._obs_set(250), now=NOW, fetch_bars_batch=batch,
                                       save_fn=lambda o: True, max_symbols=400, batch_size=100)
        self.assertEqual([len(c[0]) for c in calls], [100, 100, 50])
        self.assertEqual(sum(len(c[0]) for c in calls), len({s for c in calls for s in c[0]}))
        self.assertEqual((r["unique_symbols_requested"], r["cache_misses"], r["cache_hits"]),
                         (250, 250, 500))
        self.assertEqual(r["fetch_mode"], "batch")
        self.assertEqual(r["symbols_processed"], 250)
        self.assertGreater(r["outcomes_matured"], 0)
        self.assertEqual(r["outcomes_matured"], r["attached"])
        # window anchored at earliest observation, bounded by now
        self.assertEqual(calls[0][1], "2026-08-25T14:00:00Z")
        self.assertEqual(calls[0][2], "2026-08-25T18:00:00Z")

    def test_rate_limited_is_not_price_data_unavailable_and_breaks_circuit(self):
        calls = []

        class Throttled(RuntimeError):
            rate_limited = True

        def batch(symbols, start, end):
            calls.append(symbols)
            raise Throttled("429")
        saved = []
        r = worker.mature_observations(self._obs_set(250), now=NOW, fetch_bars_batch=batch,
                                       save_fn=saved.append, batch_size=100)
        self.assertEqual(len(calls), 1)  # circuit breaker: no hammering after exhaustion
        self.assertEqual(r["rate_limited_symbols"], 250)
        self.assertEqual(r["price_data_failures"], 0)
        self.assertNotIn("PRICE_DATA_UNAVAILABLE", r["failures"])
        self.assertEqual(r["failures"]["RATE_LIMITED"], 250)
        self.assertEqual(saved, [])

    def test_missing_data_vs_provider_error_distinguished(self):
        def batch(symbols, start, end):
            if "S000" in symbols:
                return {"S000": _bars()}  # S001 absent → true missing data
            raise ValueError("boom")
        r = worker.mature_observations(self._obs_set(3, 1), now=NOW, fetch_bars_batch=batch,
                                       save_fn=lambda o: True, batch_size=2)
        self.assertEqual(r["price_data_unavailable_symbols"], 1)
        self.assertEqual(r["provider_error_symbols"], 1)
        self.assertEqual(r["rate_limited_symbols"], 0)
        self.assertEqual(r["price_data_failures"], 4)  # 4 horizons of S001

    def test_ineligible_symbols_never_fetched_or_capped(self):
        fetched = []
        obs = [_obs("PSA.PRF"), _obs("PSEC-PRA"), _obs("NVDA")]
        r = worker.mature_observations(
            obs, now=NOW, save_fn=lambda o: True, max_symbols=1,
            fetch_bars_batch=lambda s, a, b: fetched.extend(s) or {x: _bars() for x in s})
        self.assertEqual(fetched, ["NVDA"])
        self.assertEqual(r["ineligible"]["symbols"], 2)
        self.assertEqual(r["ineligible"]["reasons"], {"preferred_share": 2})
        self.assertEqual(r["symbols_deferred"], 0)

    def test_stats_surface_in_report(self):
        stats = pa.AlpacaRequestStats()
        stats.requests, stats.rate_limited, stats.retries = 7, 2, 3
        r = worker.mature_observations(self._obs_set(2), now=NOW, request_stats=stats,
                                       fetch_bars_batch=lambda s, a, b: {x: _bars() for x in s},
                                       save_fn=lambda o: True)
        self.assertEqual((r["alpaca_requests"], r["alpaca_429_count"], r["alpaca_retry_count"]),
                         (7, 2, 3))
        self.assertEqual(r["requests_per_processed_symbol"], 3.5)
        for k in ("alpaca_requests", "alpaca_429_count", "alpaca_retry_count",
                  "unique_symbols_requested", "cache_hits", "cache_misses",
                  "symbols_processed", "symbols_deferred", "outcomes_matured",
                  "price_data_failures", "insufficient_future_bars"):
            self.assertIn(k, r)
        self.assertIn("Retrieval (batch)", worker.render_report_text(r))

    def test_batch_path_idempotent_with_store(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            obs = self._obs_set(3)
            for o in obs:
                self.assertTrue(store.save_observation(o, conn=conn))

            def run():
                return worker.mature_observations(
                    store.load_recent_observations(limit=100, attach_outcomes=True, conn=conn),
                    now=NOW, fetch_bars_batch=lambda s, a, b: {x: _bars() for x in s},
                    save_fn=lambda o: store.save_outcome(o, conn=conn))
            first = run()
            self.assertEqual(first["outcomes_matured"], 36)  # 9 obs × 4 horizons
            second = run()
            self.assertEqual(second["outcomes_matured"], 0)
            self.assertEqual(second["unique_symbols_requested"], 0)  # nothing re-fetched
            n = conn.execute("SELECT COUNT(*) FROM hsf_observation_outcomes").fetchone()[0]
            self.assertEqual(n, 36)
        finally:
            conn.close()

    def test_batch_and_legacy_paths_produce_identical_outcomes(self):
        obs = self._obs_set(4)
        a, b = [], []
        worker.mature_observations(obs, now=NOW, save_fn=lambda o: a.append(o) or True,
                                   fetch_bars=lambda s, d: _bars())
        worker.mature_observations(obs, now=NOW, save_fn=lambda o: b.append(o) or True,
                                   fetch_bars_batch=lambda s, x, y: {z: _bars() for z in s})

        def key(o):
            return (o["observation_id"], o["horizon"])
        self.assertEqual(sorted(a, key=key), sorted(b, key=key))


class RetirementTests(unittest.TestCase):
    OLD = "2026-08-18T14:00:00+00:00"   # 7 days before NOW > 6-day threshold
    RECENT = "2026-08-21T14:00:00+00:00"  # 4 days before NOW: still retried

    def test_closed_window_observations_are_retired_not_fetched(self):
        fetched = []
        obs = [_obs("OLDX", self.OLD), _obs("MIXD", self.OLD), _obs("MIXD", self.RECENT),
               _obs("NVDA", "2026-08-25T14:00:00+00:00")]
        r = worker.mature_observations(
            obs, now=NOW, save_fn=lambda o: True,
            fetch_bars_batch=lambda s, a, b: fetched.extend(s) or {x: _bars() for x in s})
        self.assertEqual(sorted(fetched), ["MIXD", "NVDA"])
        self.assertEqual(r["retired"], {"observations": 2, "horizons": 8, "symbols": 1,
                                        "retire_after_days": 6.0})
        self.assertEqual(r["backlog"]["ready_symbols"], 2)
        self.assertEqual(r["eligible_observations"], 2)
        self.assertEqual(r["failures"], {})  # retirement is not a failure
        self.assertIn("Retired (window closed > 6.0d)", worker.render_report_text(r))

    def test_already_matured_horizons_are_not_counted_as_retired(self):
        o = _obs("OLDX", self.OLD)
        o["outcomes"] = {h: True for h in oc.HORIZON_BARS}
        r = worker.mature_observations([o], now=NOW, save_fn=lambda x: True,
                                       fetch_bars_batch=lambda s, a, b: {})
        self.assertEqual(r["retired"]["observations"], 0)

    def test_retirement_disabled_reattempts_for_backfill(self):
        fetched = []
        r = worker.mature_observations(
            [_obs("OLDX", self.OLD)], now=NOW, save_fn=lambda o: True, retire_after=None,
            fetch_bars_batch=lambda s, a, b: fetched.extend(s) or {})
        self.assertEqual(fetched, ["OLDX"])
        self.assertEqual(r["retired"]["observations"], 0)
        self.assertIsNone(r["retired"]["retire_after_days"])


class SymbolExclusionTests(unittest.TestCase):
    def test_reasons(self):
        from data.us_market_universe import symbol_exclusion_reason as f
        self.assertEqual(f("PSA.PRF"), "preferred_share")
        self.assertEqual(f(""), "malformed_symbol")
        for common in ("NVDA", "MU", "SNOW", "PRE", "BRK-B", "SPY"):
            self.assertIsNone(f(common), common)


if __name__ == "__main__":
    unittest.main()
