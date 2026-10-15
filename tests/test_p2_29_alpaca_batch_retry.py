"""P2-29 — a daily-bar batch that times out (or gets a 429/5xx) is retried once,
so its 150 symbols aren't dropped from the scan; a second failure still skips
the batch without failing the scan."""
import importlib.util
import unittest
from unittest import mock

HAS_REQUESTS = importlib.util.find_spec("requests") is not None
if HAS_REQUESTS:
    import requests

    from data import price_alpaca as pa

CFG = {"api_key": "k", "api_secret": "s", "data_url": "https://data.example"}
BARS = {"bars": {"AAPL": [{"t": "2026-10-01T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]},
        "next_page_token": None}


def ok():
    r = mock.Mock(status_code=200)
    r.json.return_value = BARS
    r.raise_for_status.return_value = None
    return r


def status(code):
    r = mock.Mock(status_code=code, headers={})
    r.raise_for_status.side_effect = requests.HTTPError(f"{code}", response=r)
    return r


@unittest.skipUnless(HAS_REQUESTS, "requests not installed in this environment")
class BatchRetryTests(unittest.TestCase):
    def fetch(self, responses):
        with (
            mock.patch.object(pa, "get_alpaca_config", return_value=CFG),
            mock.patch.object(pa.requests, "get", side_effect=responses) as get,
            mock.patch.object(pa.time, "sleep") as sleep,
        ):
            out = pa.download_multi_alpaca(["AAPL"], "60d", "1d", False, 5.0, feed="iex")
        return out, get, sleep

    def test_timeout_then_success_keeps_the_batch(self):
        out, get, sleep = self.fetch([requests.Timeout("slow"), ok()])
        self.assertIn("AAPL", out)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once()
        self.assertLessEqual(sleep.call_args[0][0], 2.0)

    def test_server_error_then_success(self):
        out, get, _ = self.fetch([status(503), ok()])
        self.assertIn("AAPL", out)
        self.assertEqual(get.call_count, 2)

    def test_two_timeouts_skip_the_batch_without_raising(self):
        out, get, _ = self.fetch([requests.Timeout("slow"), requests.Timeout("slow")])
        self.assertEqual(out, {})
        self.assertEqual(get.call_count, 2)

    def test_persistent_429_skips_the_batch_without_raising(self):
        out, get, _ = self.fetch([status(429), status(429)])
        self.assertEqual(out, {})
        self.assertEqual(get.call_count, 2)

    def test_client_error_is_not_retried(self):
        out, get, sleep = self.fetch([status(400)])
        self.assertEqual(out, {})
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()


def bars_for(params):
    syms = params["symbols"].split(",")
    r = mock.Mock(status_code=200)
    r.json.return_value = {"bars": {s: [{"t": "2026-10-01T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5,
                                         "v": 100}] for s in syms}, "next_page_token": None}
    r.raise_for_status.return_value = None
    return r


@unittest.skipUnless(HAS_REQUESTS, "requests not installed in this environment")
class BatchSplitTests(unittest.TestCase):
    """2026-10-14 13:35 UTC scan: two 150-symbol batches (WMT, VZ, WFC...) timed out
    twice and were dropped, then reported as policy skips ("Timeouts: 0")."""

    SYMS = [f"S{i:03d}" for i in range(150)]

    def fetch(self, get, symbols=None):
        failures = {}
        with (
            mock.patch.object(pa, "get_alpaca_config", return_value=CFG),
            mock.patch.object(pa.requests, "get", side_effect=get) as g,
            mock.patch.object(pa.time, "sleep"),
        ):
            out = pa.download_multi_alpaca(symbols or self.SYMS, "60d", "1d", False, 5.0, feed="iex",
                                           failures=failures)
        return out, failures, g

    def test_failed_batch_is_re_requested_in_pieces(self):
        def get(url, headers, params, timeout):
            if len(params["symbols"].split(",")) == 150:
                raise requests.Timeout("Read timed out. (read timeout=10.0)")
            return bars_for(params)

        out, failures, g = self.fetch(get)
        self.assertEqual(set(out), set(self.SYMS))
        self.assertEqual(failures, {})
        self.assertEqual(g.call_count, 2 + 3)  # batch + its retry, then 3 pieces of 50

    def test_piece_that_still_fails_is_reported_as_timeout(self):
        def get(url, headers, params, timeout):
            syms = params["symbols"].split(",")
            if len(syms) == 150 or "S000" in syms:
                raise requests.Timeout("Read timed out. (read timeout=10.0)")
            return bars_for(params)

        out, failures, _ = self.fetch(get)
        self.assertEqual(len(out), 100)
        self.assertEqual(len(failures), 50)
        self.assertIn("S000", failures)
        self.assertTrue(all(r.startswith("alpaca_timeout:") for r in failures.values()))

    def test_rate_limit_reason(self):
        out, failures, _ = self.fetch([status(429), status(429)], symbols=["AAPL"])
        self.assertEqual(out, {})
        self.assertTrue(failures["AAPL"].startswith("alpaca_rate_limited:"))

    def test_class_share_failure_maps_back_to_dash_symbol(self):
        out, failures, _ = self.fetch([requests.Timeout("slow")] * 2, symbols=["BRK-B"])
        self.assertIn("BRK-B", failures)

    def test_split_budget_bounds_an_outage(self):
        def get(url, headers, params, timeout):
            raise requests.Timeout("Read timed out.")

        syms = [f"S{i:04d}" for i in range(150 * 10)]
        out, failures, g = self.fetch(get, symbols=syms)
        self.assertEqual(out, {})
        self.assertEqual(len(failures), len(syms))
        # 10 batches x 2 tries, plus 3 pieces x 2 tries for the first 6 batches only
        self.assertEqual(g.call_count, 10 * 2 + pa._MAX_SPLIT_BATCHES * 3 * 2)

    def test_no_bars_is_not_a_failure(self):
        r = mock.Mock(status_code=200)
        r.json.return_value = {"bars": {}, "next_page_token": None}
        r.raise_for_status.return_value = None
        out, failures, _ = self.fetch([r], symbols=["ZZZZ"])
        self.assertEqual((out, failures), ({}, {}))


class SkipReasonTests(unittest.TestCase):
    def test_dropped_symbols_count_as_timeouts_not_policy_skips(self):
        from unittest.mock import MagicMock, patch

        import data.prices as prices
        from analytics.coverage import classify_failure

        def fake_download(tickers, *a, failures=None, **k):
            failures["BBB"] = "alpaca_timeout:ReadTimeout"
            return {}

        prices.clear_price_cache()
        cfg = prices.PriceFetchConfig(tickers=["AAA", "BBB"])
        with (
            patch.object(prices, "_get_alpaca_config", return_value={"api_key": "key"}),
            patch.object(prices, "_download_multi_alpaca", side_effect=fake_download),
            patch.object(prices, "_yf", MagicMock()),
            patch.dict("os.environ", {"PRICE_SKIP_YF_FALLBACK": "1"}, clear=False),
        ):
            _data, skipped = prices._download_batch(["AAA", "BBB"], cfg)
        reasons = dict(skipped)
        self.assertEqual(classify_failure(reasons["BBB"]), "TIMEOUT")
        self.assertEqual(classify_failure(reasons["AAA"]), "FILTERED_BY_POLICY")
        prices.clear_price_cache()


if __name__ == "__main__":
    unittest.main()
