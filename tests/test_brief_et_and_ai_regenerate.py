"""Market Brief "As of" line uses New York time; AI summary shows Regenerate
right after generating (owner phone screenshots, 2026-09-30)."""
import datetime as dt
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True


class EtStampTests(unittest.TestCase):
    def test_naive_utc_and_aware_values(self):
        from ui.market_brief import _et_stamp

        self.assertEqual(_et_stamp(dt.datetime(2026, 9, 30, 16, 38)), "Sep 30, 12:38 PM ET")
        self.assertEqual(_et_stamp(dt.datetime(2026, 9, 30, 16, 38, tzinfo=dt.timezone.utc)), "Sep 30, 12:38 PM ET")
        self.assertEqual(_et_stamp(dt.datetime(2026, 12, 1, 14, 5, tzinfo=dt.timezone.utc)), "Dec 1, 9:05 AM ET")  # EST
        self.assertIsNone(_et_stamp(None))
        self.assertIsNone(_et_stamp("2026-09-30"))

    def test_brief_no_longer_prints_utc(self):
        src = (ROOT / "ui" / "market_brief.py").read_text()
        self.assertNotIn("UTC (latest scan snapshot)", src)
        self.assertIn('st.caption(f"📸 As of {stamp} (latest scan snapshot).")', src)


SCRIPT = '''
import pandas as pd
from utils.st_compat import st
from ui.ai_summary import render_ai_summary
render_ai_summary(pd.DataFrame([{"Ticker": "AAA", "BreakoutScore": 50.0}]), context="t")
'''


if __name__ == "__main__":
    unittest.main()
