"""Impact replay must pair the same picks and keep old policy distinct."""
import datetime as dt
import unittest

try:
    import pandas as pd
except ImportError:
    pd = None

from scripts import historical_impact_audit as audit

UTC = dt.timezone.utc


@unittest.skipIf(pd is None, 'needs pandas')
class ImpactAuditTests(unittest.TestCase):
    def test_paired_policy_delta_uses_same_symbol_and_exit(self):
        index = pd.bdate_range('2026-10-05', periods=6)
        bars = {'AAA': pd.DataFrame({'Open': [100, 110, 110, 110, 110, 110],
                                     'Close': [100, 100, 100, 100, 100, 110]}, index=index),
                'SPY': pd.DataFrame({'Open': [100]*6, 'Close': [100]*6}, index=index)}
        signal = dt.datetime(2026, 10, 5, 19, tzinfo=UTC)
        result = audit.compare([(signal, ['AAA'])],bars,5,dt.datetime(2026,11,6,tzinfo=UTC))
        self.assertEqual(result['paired_legacy']['n'],1)
        self.assertAlmostEqual(result['paired_legacy']['avg_excess_return'],.1)
        self.assertAlmostEqual(result['paired_corrected']['avg_excess_return'],0)
        self.assertAlmostEqual(result['paired_excess_delta_pp'],-10)
        self.assertEqual(result['counts']['same_exit_session_paired'],1)

    def test_post_close_legacy_entry_is_flagged_and_open_fallback_counted(self):
        bars = {'AAA': pd.DataFrame({'Close':[100,105]}, index=pd.bdate_range('2026-10-05',periods=2)),
                'SPY': pd.DataFrame({'Open':[100,100], 'Close':[100,100]}, index=pd.bdate_range('2026-10-05',periods=2))}
        result = audit.compare([(dt.datetime(2026,10,5,22,tzinfo=UTC),['AAA'])],bars,1,dt.datetime(2026,11,6,tzinfo=UTC))
        self.assertEqual(result['counts']['legacy_entry_before_snapshot'],1)
        self.assertEqual(result['counts']['legacy_open_fallback_picks'],1)
        self.assertEqual(result['paired_corrected']['n'],0)
        self.assertEqual(result['corrected_exclusions'],{'ENTRY_FIELD_UNAVAILABLE':1})

    def test_naive_database_timestamp_reproduces_new_reader_empty_result(self):
        raw = dt.datetime(2026,11,5,20)
        rows = [{'public':True,'stored_created_at':raw,'signal_at':raw.replace(tzinfo=UTC)}]
        now = dt.datetime(2026,11,6,23,tzinfo=UTC)
        self.assertEqual(audit.current_select(rows,now,raw_timestamps=True),[])
        self.assertEqual(len(audit.current_select(rows,now)),1)

    def test_private_rows_are_never_selected_for_new_replay(self):
        stamp = dt.datetime(2026,11,5,20,tzinfo=UTC)
        rows = [{'public':False,'stored_created_at':stamp.replace(tzinfo=None),'signal_at':stamp}]
        self.assertEqual(audit.current_select(rows,dt.datetime(2026,11,6,23,tzinfo=UTC)),[])
