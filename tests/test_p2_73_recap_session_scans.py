"""P2-73 — the recap line names the pre-market and after-hours session scans next
to the full-market count, so '3 full-market scans' doesn't read as missing runs."""
import datetime as dt
import unittest

from ui import recap

FRI = dt.date(2026, 10, 2)


def run(label, created):
    return {"id": 1, "label": label, "created_at": created}


class SessionCountTests(unittest.TestCase):
    RUNS = [run("premarket", "2026-10-02T12:35:00+00:00"),     # Fri 8:35 ET
            run("postmarket", "2026-10-02T20:35:00+00:00"),
            run("postmarket", "2026-10-02T21:35:00+00:00"),
            run("postmarket", "2026-10-03T02:56:00+00:00"),    # Fri 10:56 PM ET, same ET day
            run("premarket", "2026-10-01T12:35:00+00:00"),     # Thursday
            run("US_MARKET", "2026-10-02T13:35:00+00:00")]

    def test_counts_by_et_day(self):
        self.assertEqual(recap.session_scan_counts(self.RUNS, FRI), {"premarket": 1, "postmarket": 3})
        self.assertEqual(recap.session_scan_counts([], FRI), {"premarket": 0, "postmarket": 0})


class LineTests(unittest.TestCase):
    def line(self, **kw):
        r = {"scans": 3, "entered": [], "left": [], "standouts": [], **kw}
        return recap.recap_lines(r)[0]

    def test_both_sessions(self):
        self.assertEqual(self.line(premarket_scans=1, postmarket_scans=2),
                         "- **3** full-market scans ran (plus 1 pre-market and 2 after-hours).")

    def test_one_session_only(self):
        self.assertEqual(self.line(premarket_scans=1, postmarket_scans=0),
                         "- **3** full-market scans ran (plus 1 pre-market).")

    def test_no_session_scans_keeps_the_old_line(self):
        self.assertEqual(self.line(), "- **3** full-market scans ran.")


if __name__ == "__main__":
    unittest.main()
