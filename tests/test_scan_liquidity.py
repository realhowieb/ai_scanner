"""scan.liquidity: pre-download trim for Combo / US market custom scans."""
import importlib.util
import unittest
from unittest import mock

from scan import liquidity as liq


def snap(price, vol_today=0, vol_prev=0, prev_price=None):
    prev_price = prev_price or price
    return {"latestTrade": {"p": price},
            "dailyBar": {"c": price, "v": vol_today},
            "prevDailyBar": {"c": prev_price, "v": vol_prev}}


class KeepRuleTests(unittest.TestCase):
    def keep(self, s, lo=5.0, hi=500.0, floor=5_000_000):
        return liq.keep_symbol(s, min_price=lo, max_price=hi, min_avg_dollar_vol=floor)

    def test_price_band_with_margin(self):
        self.assertTrue(self.keep(snap(4.6, 2_000_000)))     # within 10% below the $5 floor
        self.assertFalse(self.keep(snap(4.4, 2_000_000)))
        self.assertTrue(self.keep(snap(540, 10_000)))         # within 10% above $500
        self.assertFalse(self.keep(snap(560, 10_000)))

    def test_dollar_volume_quarter_of_floor_on_either_day(self):
        self.assertTrue(self.keep(snap(10, vol_today=125_000)))          # $1.25M = 1/4 of $5M
        self.assertFalse(self.keep(snap(10, vol_today=120_000)))
        self.assertTrue(self.keep(snap(10, vol_today=1_000, vol_prev=200_000)))  # quiet today, normal yesterday

    def test_no_floor_means_price_only(self):
        self.assertTrue(self.keep(snap(10, 0), floor=0))

    def test_no_price_is_dropped(self):
        self.assertFalse(self.keep({"dailyBar": {"v": 10_000_000}}))

    def test_price_falls_back_to_bars(self):
        self.assertTrue(self.keep({"dailyBar": {}, "prevDailyBar": {"c": 20, "v": 1_000_000}}))


class BatchTests(unittest.TestCase):
    def test_trims_keeps_order_and_reports(self):
        data = {"AAA": snap(10, 1_000_000), "BBB": snap(1, 10_000_000), "CCC": snap(50, 500_000)}
        stats = {}
        out = liq.apply_liquidity_filter_batch(["ccc", "bbb", "zzz", "aaa", "AAA"], min_price=5,
                                               min_avg_dollar_vol=5_000_000, max_price=500,
                                               fetch=lambda batch: {s: data[s] for s in batch if s in data},
                                               stats=stats)
        self.assertEqual(out, ["CCC", "AAA"])
        self.assertEqual(stats, {"requested": 4, "kept": 2, "failed_batches": 0,
                                 "dropped_no_data": 1, "dropped_price_or_volume": 1})

    def test_failed_batch_keeps_its_symbols(self):
        syms = [f"S{i:04d}" for i in range(450)]          # 3 batches of 200/200/50
        calls = []

        def fetch(batch):
            calls.append(len(batch))
            if batch[0] == "S0200":
                return None                                # second batch fails
            return {}                                      # others: no data for anyone
        out = liq.apply_liquidity_filter_batch(syms, min_price=5, min_avg_dollar_vol=0, fetch=fetch)
        self.assertEqual(sorted(calls), [50, 200, 200])
        self.assertEqual(out, syms[200:400])

    def test_everything_failing_keeps_the_whole_list(self):
        syms = ["AAA", "BBB"]
        self.assertEqual(liq.apply_liquidity_filter_batch(syms, min_price=5, min_avg_dollar_vol=1,
                                                          fetch=lambda b: None), syms)

    def test_no_alpaca_configuration_keeps_the_list(self):
        with mock.patch.object(liq, "_alpaca_fetch", return_value=None):
            self.assertEqual(liq.apply_liquidity_filter_batch(["A", "B"], min_price=5, min_avg_dollar_vol=1),
                             ["A", "B"])

    def test_empty(self):
        self.assertEqual(liq.apply_liquidity_filter_batch([], min_price=5, min_avg_dollar_vol=1), [])


@unittest.skipUnless(importlib.util.find_spec("requests"), "needs requests")
class AlpacaFetchTests(unittest.TestCase):
    def test_class_shares_and_feed(self):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"BRK.B": snap(400, 1_000_000), "AAPL": snap(200, 1)}
        with mock.patch("data.alpaca_config.get_alpaca_config",
                        return_value={"data_url": "https://data.example", "base_url": "x",
                                      "api_key": "k", "api_secret": "s"}), \
                mock.patch("data.alpaca_config.get_alpaca_headers", return_value={"h": "1"}), \
                mock.patch("data.alpaca_config.get_alpaca_data_feed", return_value="sip"), \
                mock.patch("requests.get", return_value=resp) as get:
            fetch = liq._alpaca_fetch()
            out = fetch(["BRK-B", "AAPL"])
        self.assertEqual(set(out), {"BRK-B", "AAPL"})
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["symbols"], "BRK.B,AAPL")
        self.assertEqual(params["feed"], "sip")
        self.assertEqual(get.call_args.args[0], "https://data.example/v2/stocks/snapshots")

    def test_http_error_is_a_failed_batch(self):
        with mock.patch("data.alpaca_config.get_alpaca_config",
                        return_value={"data_url": "https://d", "base_url": "x", "api_key": "k", "api_secret": "s"}), \
                mock.patch("data.alpaca_config.get_alpaca_headers", return_value={"h": "1"}), \
                mock.patch("requests.get", return_value=mock.Mock(status_code=429)):
            self.assertIsNone(liq._alpaca_fetch()(["AAPL"]))


if __name__ == "__main__":
    unittest.main()
