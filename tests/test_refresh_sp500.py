"""SPY-holdings S&P 500 refresh — stdlib xlsx parse + safety guards."""
import io
import unittest
import zipfile
from unittest import mock

from scripts import refresh_sp500 as r


def _make_xlsx(rows: list[list[str]]) -> bytes:
    """Build a minimal .xlsx (inline strings) with the given rows/cells."""
    def col(n):  # 0 -> A, 1 -> B ...
        s = ""
        n += 1
        while n:
            n, rem = divmod(n - 1, 26)
            s = chr(65 + rem) + s
        return s

    cells_xml = []
    for ri, row in enumerate(rows, start=1):
        cs = "".join(
            f'<c r="{col(ci)}{ri}" t="inlineStr"><is><t>{val}</t></is></c>'
            for ci, val in enumerate(row)
        )
        cells_xml.append(f'<row r="{ri}">{cs}</row>')
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = (f'<?xml version="1.0"?><worksheet xmlns="{ns}"><sheetData>'
             + "".join(cells_xml) + "</sheetData></worksheet>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return buf.getvalue()


class ParseTests(unittest.TestCase):
    def _holdings(self, tickers):
        rows = [
            ["Fund", "SPDR SP500"],
            ["Date", "2026-09-16"],
            [],
            ["Name", "Ticker", "Weight", "Sector"],
        ]
        for t in tickers:
            rows.append([f"{t} Inc", t, "1.0", "Tech"])
        rows.append(["US DOLLAR", "-", "0.1", "Cash"])  # non-equity row dropped
        return _make_xlsx(rows)

    def test_parses_tickers_and_drops_cash(self):
        out = r.parse_spy_tickers(self._holdings(["AAPL", "MSFT", "BRK.B"]))
        self.assertEqual(out, ["AAPL", "BRK-B", "MSFT"])  # normalized, sorted, cash dropped

    def test_missing_ticker_column_raises(self):
        xlsx = _make_xlsx([["Name", "Weight"], ["Apple", "1.0"]])
        with self.assertRaises(ValueError):
            r.parse_spy_tickers(xlsx)


class GuardTests(unittest.TestCase):
    def test_implausible_result_leaves_file_unchanged(self):
        # too few tickers / missing sentinels -> non-zero exit, no write
        path = mock.MagicMock()
        with (
            mock.patch.object(r, "fetch_spy_tickers", return_value=["AAPL", "MSFT"]),
            mock.patch.object(r, "SP500_PATH", path),
        ):
            rc = r.main()
        self.assertEqual(rc, 1)
        path.write_text.assert_not_called()

    def test_fetch_failure_leaves_file_unchanged(self):
        path = mock.MagicMock()
        with (
            mock.patch.object(r, "fetch_spy_tickers", side_effect=RuntimeError("network")),
            mock.patch.object(r, "SP500_PATH", path),
        ):
            rc = r.main()
        self.assertEqual(rc, 1)
        path.write_text.assert_not_called()

    def test_plausible_result_writes(self):
        good = sorted(set(list(r._SENTINELS) + [f"T{i:03d}" for i in range(500)]))
        path = mock.MagicMock()
        path.read_text.return_value = "AAPL\n"
        with (
            mock.patch.object(r, "fetch_spy_tickers", return_value=good),
            mock.patch.object(r, "SP500_PATH", path),
        ):
            rc = r.main()
        self.assertEqual(rc, 0)
        path.write_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
