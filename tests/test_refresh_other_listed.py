"""NYSE / other-exchange listed-directory refresh: parse + safety guards, and the
movers universe picking the file up."""
import unittest
from unittest import mock

from scripts import refresh_other_listed as r

_HEADER = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"


def _line(sym, name, exch="N", etf="N", test="N"):
    return f"{sym}|{name}|{exch}|{sym}|{etf}|100|{test}|{sym}"


class ParseTests(unittest.TestCase):
    def test_keeps_common_drops_junk(self):
        body = "\n".join([
            _HEADER,
            _line("GME", "GameStop Corporation Common Stock"),
            _line("BRK.B", "Berkshire Hathaway Inc. Class B"),
            _line("SPY", "SPDR S&P 500 ETF Trust", exch="P", etf="Y"),
            _line("ABR$D", "Arbor Realty Trust Preferred Series D"),
            _line("XYZ.U", "Some SPAC Units"),
            _line("XYZ.WS", "Some SPAC Warrants"),
            _line("PTY", "PIMCO Corporate & Income Opportunity Fund"),
            _line("TSTX", "Test Issue Inc", test="Y"),
            _line("SNOW", "Snowflake Inc. Class A Common Stock"),
            "File Creation Time: 1101202621:01||||||",
        ])
        self.assertEqual(r.parse_other_listed(body), ["BRK-B", "GME", "SNOW"])


class GuardTests(unittest.TestCase):
    def test_implausible_leaves_unchanged(self):
        path = mock.MagicMock()
        with (
            mock.patch.object(r, "fetch_other_tickers", return_value=["GME", "JPM"]),
            mock.patch.object(r, "OTHER_PATH", path),
        ):
            self.assertEqual(r.main(), 1)
        path.write_text.assert_not_called()

    def test_fetch_failure_leaves_unchanged(self):
        path = mock.MagicMock()
        with (
            mock.patch.object(r, "fetch_other_tickers", side_effect=RuntimeError("net")),
            mock.patch.object(r, "OTHER_PATH", path),
        ):
            self.assertEqual(r.main(), 1)
        path.write_text.assert_not_called()

    def test_plausible_writes(self):
        tickers = sorted({*r._SENTINELS, *(f"Z{i:04d}" for i in range(r._MIN_TICKERS))})
        path = mock.MagicMock()
        path.exists.return_value = False
        with (
            mock.patch.object(r, "fetch_other_tickers", return_value=tickers),
            mock.patch.object(r, "OTHER_PATH", path),
        ):
            self.assertEqual(r.main(), 0)
        path.write_text.assert_called_once()


class MoversUniverseTests(unittest.TestCase):
    def test_movers_universe_adds_other_exchanges(self):
        from ui import day_trader as dtm

        with (
            mock.patch.object(dtm, "_sp500_universe", return_value=["AAPL", "JPM"]),
            mock.patch.object(dtm, "_nasdaq_universe", return_value=["AAPL", "DKNG"]),
            mock.patch.object(dtm, "_other_listed_universe", return_value=["GME", "JPM"]),
        ):
            self.assertEqual(dtm._movers_universe(), ["AAPL", "JPM", "DKNG", "GME"])

    def test_missing_file_reads_empty(self):
        from ui import day_trader as dtm

        with mock.patch("pathlib.Path.exists", return_value=False):
            self.assertEqual(dtm._other_listed_universe(), [])


if __name__ == "__main__":
    unittest.main()
