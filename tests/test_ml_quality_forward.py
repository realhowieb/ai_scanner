"""No future data in quality diagnostics; prospective collection has fixed rules."""
import datetime as dt
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analytics import ml_development_quality as Q
from db import ml_forward_cohorts as F

ROOT=Path(__file__).resolve().parents[1]


class QualityTests(unittest.TestCase):
    def test_missing_nonfinite_zero_and_constant_are_distinct(self):
        result=Q.describe([None,float('nan'),float('inf'),0,0])
        self.assertEqual(result['missing'],3)
        self.assertEqual(result['unique'],1)
        self.assertEqual(result['zero_pct_of_observed'],100)
        self.assertEqual(Q.describe([None])['median'],None)

    def test_psi_detects_shift_and_missingness_without_fitting_on_late_values(self):
        early=list(range(100))
        self.assertEqual(Q.psi(early,early),0)
        self.assertGreater(Q.psi(early,[1000]*100),1)
        self.assertGreater(Q.psi(early,[None]*100),1)
        self.assertIsNone(Q.psi([None],[5]))

    def test_time_split_and_provenance_masks_are_honest(self):
        rows=[SimpleNamespace(observation_id=i,entry_day=dt.date(2026,9,14)+dt.timedelta(days=i),
               features={'x':0 if i<4 else None,'setup':'new' if i>=4 else 'old'},labels={'mfe_5d':.05 if i<4 else 0,'mae_5d':-.01}) for i in range(8)]
        raw=[{'id':0,'raw_signal':{'models':{'prebreakout':{'status':'captured','feature_names':['x'],'default_mask':[True]}}}}]
        report=Q.report(rows,raw,numeric_features=['x'],categorical_features=['setup'])
        self.assertEqual(report['features']['x']['early']['missing'],0)
        self.assertEqual(report['features']['x']['late']['missing'],4)
        self.assertEqual(report['categorical']['setup']['late_unseen_pct'],100)
        self.assertNotIn('unrecorded',report['provenance']['prebreakout']['status_counts'])
        self.assertEqual(report['provenance']['prebreakout']['features']['x']['defaulted'],1)

    def test_recorded_values_are_crosschecked_without_filling_missing_features(self):
        rows=[SimpleNamespace(observation_id=i,entry_day=dt.date(2026,9,14)+dt.timedelta(days=i),
              features={'trend_10d_pct':None},labels={'mfe_5d':0,'mae_5d':0}) for i in range(4)]
        def raw(i,mask):
            return {'id':i,'raw_signal':{'models':{'prebreakout':{'status':'captured',
                   'feature_names':['Trend10D%'],'feature_values':[0],'default_mask':[mask]}}}}
        result=Q.report(rows,[raw(0,False),raw(1,True),raw(2,None)],numeric_features=['trend_10d_pct'],categorical_features=[])
        self.assertEqual(result['recorded_input_crosscheck']['trend_10d_pct']['missing_with_recorded_observed_value'],1)
        self.assertTrue(all(r.features['trend_10d_pct'] is None for r in rows))

    def test_fixed_protocol_has_twenty_sessions(self):
        protocol=json.loads((ROOT/'docs/ml_forward_cohort_2026-12-23.json').read_text())
        start,end,sessions=F.validate(protocol)
        self.assertEqual(len(sessions),20)
        self.assertNotIn('2026-12-25',sessions)
        self.assertEqual(sessions[0],'2026-12-23')
        self.assertEqual(sessions[-1],'2027-01-22')
        self.assertLess(start,end)

    def test_scope_rejects_outside_boundary_before_building_features(self):
        from scripts import ml_development_quality as script
        entry={'observation_ids':[1],'created_at':'2026-11-13T16:16:00Z'}
        scope={'dataset_version':'ds','development_start':'2026-09-14','development_end_exclusive':'2026-10-22'}
        with patch('db.research_datasets.get_dataset_version',return_value=entry), \
             patch('db.ml_forward_cohorts.assert_development_allowed'), \
             patch('db.research_datasets.fetch_opportunity_rows',return_value=[{'id':1,'fired_at':'2026-11-01T12:00:00Z'}]), \
             patch('db.research_datasets.fetch_scan_records') as scans:
            with self.assertRaisesRegex(ValueError,'outside the development'):
                script.load_scope(scope)
        scans.assert_not_called()
