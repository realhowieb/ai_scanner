"""P2-29 — a daily-bar batch that times out (or gets a 429/5xx) is retried once,
so its 150 symbols aren't dropped from the scan; a second failure still skips
the batch without failing the scan."""
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
