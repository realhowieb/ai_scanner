"""Nasdaq listed-directory refresh — parse + safety guards."""
import unittest
from unittest import mock

from scripts import refresh_nasdaq as r

_HEADER = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"


def _line(sym, name, test="N", etf="N"):
    return f"{sym}|{name}|Q|{test}|N|100|{etf}|N"


class ParseTests(unittest.TestCase):
    def test_keeps_common_drops_junk(self):
        body = "\n".join([
            _HEADER,
            _line("AAPL", "Apple Inc. - Common Stock"),
            _line("ABCDW", "SPAC Corp - Warrant"),          # name says Warrant
            _line("ABCDU", "SPAC Corp - Units"),            # name says Units
            _line("XYZZU", "Something - Common Stock"),      # 5-char ending U -> dropped
            _line("TSTT", "Test Co", test="Y"),              # test issue
            _line("SPY", "SPDR ETF", etf="Y"),               # ETF
            _line("NVDA", "NVIDIA Corp - Common Stock"),
            "File Creation Time: 2026-09-17",
        ])
        out = r.parse_nasdaq_listed(body)
        self.assertEqual(out, ["AAPL", "NVDA"])

    def test_five_char_non_uwr_kept(self):
        body = "\n".join([_HEADER, _line("GOOGL", "Alphabet Inc. - Class A")])
        self.assertEqual(r.parse_nasdaq_listed(body), ["GOOGL"])


class GuardTests(unittest.TestCase):
    def test_implausible_leaves_unchanged(self):
        path = mock.MagicMock()
        with (
            mock.patch.object(r, "fetch_nasdaq_tickers", return_value=["AAPL", "MSFT"]),
            mock.patch.object(r, "NASDAQ_PATH", path),
        ):
            rc = r.main()
        self.assertEqual(rc, 1)
        path.write_text.assert_not_called()

    def test_fetch_failure_leaves_unchanged(self):
        path = mock.MagicMock()
        with (
            mock.patch.object(r, "fetch_nasdaq_tickers", side_effect=RuntimeError("net")),
            mock.patch.object(r, "NASDAQ_PATH", path),
        ):
            rc = r.main()
        self.assertEqual(rc, 1)
        path.write_text.assert_not_called()

    def test_plausible_writes(self):
        good = sorted(set(list(r._SENTINELS) + [f"T{i:03d}" for i in range(2100)]))
        path = mock.MagicMock()
        path.read_text.return_value = "AAPL\n"
        with (
            mock.patch.object(r, "fetch_nasdaq_tickers", return_value=good),
            mock.patch.object(r, "NASDAQ_PATH", path),
        ):
            rc = r.main()
        self.assertEqual(rc, 0)
        path.write_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
