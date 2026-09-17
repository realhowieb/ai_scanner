"""Live tradability filter — drops delisted/non-tradable symbols, fails open."""
import unittest
from unittest import mock

from data import tradability as tr


def _reset_cache():
    tr._CACHE["symbols"] = None
    tr._CACHE["fetched_at"] = 0.0


class FilterTests(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def test_drops_untradable_keeps_order(self):
        active = {f"S{i}" for i in range(600)} | {"NVDA", "AMD"}  # EA absent -> delisted
        with mock.patch.object(tr, "active_tradable_symbols", return_value=active):
            out = tr.filter_tradable_tickers(["NVDA", "EA", "AMD"])
        self.assertEqual(out, ["NVDA", "AMD"])  # EA dropped, order preserved

    def test_fails_open_when_unavailable(self):
        with mock.patch.object(tr, "active_tradable_symbols", return_value=None):
            out = tr.filter_tradable_tickers(["NVDA", "EA", "AMD"])
        self.assertEqual(out, ["NVDA", "EA", "AMD"])  # never empties a scan

    def test_normalizes_input(self):
        active = {f"S{i}" for i in range(600)} | {"NVDA"}
        with mock.patch.object(tr, "active_tradable_symbols", return_value=active):
            out = tr.filter_tradable_tickers([" nvda ", "ea"])
        self.assertEqual(out, ["NVDA"])

    def test_empty_input(self):
        self.assertEqual(tr.filter_tradable_tickers([]), [])


class FetchTests(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def _fake_requests(self, status=200, payload=None):
        fake = mock.MagicMock()
        resp = mock.MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        fake.get.return_value = resp
        return fake

    def _run_fetch(self, fake_requests):
        with (
            mock.patch.dict("sys.modules", {"requests": fake_requests}),
            mock.patch("data.alpaca_config.get_alpaca_config",
                       return_value={"base_url": "https://paper-api.alpaca.markets"}),
            mock.patch("data.alpaca_config.get_alpaca_headers", return_value={"k": "v"}),
        ):
            return tr._fetch_active_tradable_symbols()

    def test_parses_active_tradable_only(self):
        payload = [{"symbol": f"S{i}", "tradable": True, "status": "active"} for i in range(600)]
        payload += [{"symbol": "HALT", "tradable": False, "status": "active"}]
        got = self._run_fetch(self._fake_requests(payload=payload))
        self.assertIn("S1", got)
        self.assertNotIn("HALT", got)  # non-tradable excluded

    def test_rejects_suspiciously_small_response(self):
        # a tiny response is not trusted to filter the whole universe
        payload = [{"symbol": "NVDA", "tradable": True}]
        self.assertIsNone(self._run_fetch(self._fake_requests(payload=payload)))

    def test_non_200_returns_none(self):
        self.assertIsNone(self._run_fetch(self._fake_requests(status=500, payload=[])))

    def test_no_credentials_returns_none(self):
        with mock.patch("data.alpaca_config.get_alpaca_config", return_value=None):
            self.assertIsNone(tr._fetch_active_tradable_symbols())


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def test_chokepoint_drops_delisted(self):
        # the scan chokepoint filters the universe before fetching prices
        from scan import headless_common as hc
        active = {"NVDA", "AMD"} | {f"S{i}" for i in range(600)}
        with (
            mock.patch("data.tradability.active_tradable_symbols", return_value=active),
            mock.patch.object(hc, "fetch_headless_prices", return_value=({}, [], 0.0)) as fetch,
            mock.patch.object(hc, "build_filtered_price_data", return_value={}),
            mock.patch.object(hc, "maybe_run_gap_filter"),
            mock.patch.object(hc, "run_headless_breakout", return_value=__import__("pandas").DataFrame()),
        ):
            _df, meta = hc.run_headless_pipeline(
                "regular", ["NVDA", "EA", "AMD"], min_price=5, max_price=1000,
                min_dollar_vol=1, use_parallel=False, parallel_workers=1,
                parallel_chunk=10, apply_gap_filter=False, top_n=10)
        # EA was dropped before the price fetch
        fetched_symbols = fetch.call_args[0][0]
        self.assertNotIn("EA", fetched_symbols)
        self.assertEqual(meta["dropped_untradable"], 1)


if __name__ == "__main__":
    unittest.main()


class SessionUniverseTests(unittest.TestCase):
    def test_session_universe_is_sp500_plus_full_nasdaq(self):
        import os as _os

        from scan import pre_post
        with (
            mock.patch("scan.pre_post.load_sp500_tickers", return_value=["AAPL", "MSFT"]),
            mock.patch("scan.pre_post.load_nasdaq_tickers", return_value=["MSFT", "TSLA", "NVDA"]),
            mock.patch.dict(_os.environ, {"CRON_NASDAQ_LIMIT": "10000"}, clear=False),
        ):
            uni = pre_post._load_session_universe()
        self.assertEqual(uni, ["AAPL", "MSFT", "TSLA", "NVDA"])  # deduped, order preserved

    def test_session_universe_honors_limit(self):
        import os as _os

        from scan import pre_post
        with (
            mock.patch("scan.pre_post.load_sp500_tickers", return_value=["AAPL"]),
            mock.patch("scan.pre_post.load_nasdaq_tickers", return_value=["A", "B", "C", "D"]),
            mock.patch.dict(_os.environ, {"CRON_NASDAQ_LIMIT": "2"}, clear=False),
        ):
            uni = pre_post._load_session_universe()
        self.assertEqual(uni, ["AAPL", "A", "B"])  # nasdaq truncated to 2

    def test_session_universe_falls_back_without_nasdaq(self):
        from scan import pre_post
        with (
            mock.patch("scan.pre_post.load_sp500_tickers", return_value=["AAPL", "MSFT"]),
            mock.patch("scan.pre_post.load_nasdaq_tickers", return_value=[]),
        ):
            uni = pre_post._load_session_universe()
        self.assertEqual(uni, ["AAPL", "MSFT"])  # SP500 only, never empty
