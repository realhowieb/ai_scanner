"""Open in Webull links (data/broker_links.py, GET /v1/public/webull-link/{ticker}). No network."""
import unittest
from unittest import mock

from data import broker_links as bl
from tests.test_api_v1 import DEPS
from tests.test_api_v1_data import DataApiBase


class WebullUrlTests(unittest.TestCase):
    def setUp(self):
        bl._reset_cache()

    def test_exchange_prefixes(self):
        self.assertEqual(bl.webull_quote_url("AAPL", "NASDAQ"), "https://www.webull.com/quote/nasdaq-aapl")
        self.assertEqual(bl.webull_quote_url("SPY", "ARCA"), "https://www.webull.com/quote/nysearca-spy")
        self.assertEqual(bl.webull_quote_url("UEC", "AMEX"), "https://www.webull.com/quote/amex-uec")
        self.assertEqual(bl.webull_quote_url("MTUM", "BATS"), "https://www.webull.com/quote/bats-mtum")
        self.assertEqual(bl.webull_quote_url("TCEHY", "OTC"), "https://www.webull.com/quote/otcpk-tcehy")

    def test_class_shares_use_dash(self):
        self.assertEqual(bl.webull_quote_url("BRK.B", "NYSE"), "https://www.webull.com/quote/nyse-brk-b")
        self.assertEqual(bl.webull_quote_url("MOG-A", "NYSE"), "https://www.webull.com/quote/nyse-mog-a")

    def test_unknown_exchange_or_bad_ticker(self):
        self.assertIsNone(bl.webull_quote_url("AAPL", None))
        self.assertIsNone(bl.webull_quote_url("AAPL", "IEX"))
        self.assertIsNone(bl.webull_quote_url("../x", "NYSE"))

    def test_link_falls_back_to_markets_page(self):
        link = bl.webull_link("ZZZZ", fetch=lambda t: None)
        self.assertEqual(link["url"], bl.WEBULL_FALLBACK_URL)
        self.assertFalse(link["exact"])

    def test_lookup_cached_and_class_share_lookup(self):
        calls = []
        fetch = lambda t: calls.append(t) or "NYSE"
        self.assertEqual(bl.webull_link("brk-b", fetch=fetch)["url"], "https://www.webull.com/quote/nyse-brk-b")
        bl.webull_link("BRK-B", fetch=fetch)
        self.assertEqual(calls, ["BRK-B"])

    def test_alpaca_lookup_uses_dot_form(self):
        resp = mock.Mock(status_code=200, json=lambda: {"symbol": "BRK.B", "exchange": "NYSE"})
        with mock.patch("data.alpaca_config.get_alpaca_config", return_value={"base_url": "https://x"}), \
             mock.patch("data.alpaca_config.get_alpaca_headers", return_value={"k": "v"}), \
             mock.patch.dict("sys.modules", {"requests": mock.Mock(get=mock.Mock(return_value=resp))}) as mods:
            self.assertEqual(bl._fetch_exchange("BRK-B"), "NYSE")
            get = mods["requests"].get
        self.assertEqual(get.call_args[0][0], "https://x/v2/assets/BRK.B")


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class WebullLinkApiTests(DataApiBase):
    def test_signed_out_route(self):
        bl._reset_cache()
        with mock.patch.object(bl, "exchange_for", return_value="NASDAQ"):
            r = self.client.get("/v1/public/webull-link/aapl")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ticker": "AAPL", "exchange": "NASDAQ",
                                    "url": "https://www.webull.com/quote/nasdaq-aapl", "exact": True})
