"""Read development rows only; refuse a population different from the original training slice."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_scope(scope):
    from analytics import ml_v3_audit as A
    from analytics import ml_v3_pipeline as V3
    from analytics import ml_v4_providers as P
    from analytics import research_dataset as rd
    from db.ml_forward_cohorts import assert_development_allowed
    from db.research_datasets import fetch_opportunity_rows, fetch_scan_records, get_dataset_version
    from scripts.ml_v4_train import _as_list, _benchmark_computed_at

    entry = get_dataset_version(scope['dataset_version'])
    if not entry:
        raise ValueError('Frozen dataset does not exist')
    members = _as_list(entry['observation_ids'])
    # Guard membership before reading any labels. The historical dataset ends
    # before the new forward reservation, so it is allowed but never read whole.
    assert_development_allowed(members)
    lo, hi = (dt.datetime.fromisoformat(scope[k]).replace(tzinfo=dt.timezone.utc)
              for k in ('development_start','development_end_exclusive'))
    raw = fetch_opportunity_rows(lo, hi, ids=members)
    if any(not lo <= rd.to_dt(r['fired_at']) < hi for r in raw):
        raise ValueError('Reader returned a row outside the development boundary')
    bca = _benchmark_computed_at([int(r['id']) for r in raw])
    raw = V3.as_of([{**r,'benchmark_computed_at':bca.get(int(r['id']))} for r in raw],rd.to_dt(entry['created_at']))
    scans = fetch_scan_records({r['ticker'] for r in raw},lo-rd.MAX_SCAN_LAG-dt.timedelta(hours=1),hi)
    rows = [r for r,_ in A.eligible(A.build_audit_rows(rd.build_records(raw,scans),raw,unit='signal_day'),5,P.PRIMARY_TARGET.label)]
    digest = hashlib.sha256(json.dumps(sorted(r.observation_id for r in rows)).encode()).hexdigest()
    if len(rows)!=scope['expected_rows'] or digest!=scope['expected_train_ids_sha256']:
        raise ValueError('Development membership differs from original training; diagnostics refused')
    if A.slice_fingerprint(rows)!=scope['expected_slice_fingerprint']:
        raise ValueError('Development features/labels changed since training; diagnostics refused')
    return rows,raw


def main():
    from analytics import ml_development_quality as Q
    from analytics import ml_v3_audit as A
    from analytics import ml_v4_providers as P

    scope=json.loads((ROOT/'docs/ml_development_quality_scope_2026-11-13.json').read_text())
    rows,raw=load_scope(scope)
    numeric=sorted({n for fs in P.FEATURE_SETS.values() if fs.research_schema==1 for n in fs.numeric})
    result=Q.report(rows,raw,numeric_features=numeric,categorical_features=A.CATEGORICAL_FEATURES)
    result['scope']=scope
    out=ROOT/'artifacts/ml_development_quality'
    out.mkdir(parents=True,exist_ok=True)
    (out/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
