"""Offline, synthetic recovery workload; no credentials or production connections.

Run: python -m scripts.benchmark_db_transfer
Estimates application payloads using db.traffic.size, not PostgreSQL wire bytes.
"""
import datetime as dt
import json
from unittest import mock

from db.traffic import count, row_size, scope
from scripts import ml_readiness


def benchmark(row_count=10000):
    now = dt.datetime(2026, 10, 9, tzinfo=dt.timezone.utc)
    rows = [{'id': i, 'ticker': f'S{i:05d}', 'fired_at': now, 'created_at': now,
             'raw_signal': {'hsf_score': 50, 'status': 'WATCH', 'score_version': 'v1'},
             'indicators': {'status': 'WATCH'}, 'return_1d': None, 'return_3d': None, 'return_5d': None}
            for i in range(row_count)]
    scans = [{'symbol': r['ticker'], 'timestamp': now} for r in rows]
    results = []
    outputs = []
    for optimized in (False, True):
        attempts = [0]
        def read(*args):
            count(calls=1, rows_fetched=len(rows), db_to_client_payload_bytes=sum(row_size(r) for r in rows))
            return rows
        def index(*args):
            count(calls=1)
            attempts[0] += 1
            if attempts[0] == 1:
                count(errors=1)
                raise RuntimeError('synthetic transient failure')
            count(rows_fetched=len(scans), db_to_client_payload_bytes=sum(row_size(r) for r in scans))
            return scans
        with scope('benchmark.synthetic_retry') as metrics, \
             mock.patch('db.research_datasets.fetch_readiness_rows', side_effect=read), \
             mock.patch('db.research_datasets.fetch_scan_index', side_effect=index), \
             mock.patch.object(ml_readiness.time, 'sleep'), mock.patch.object(ml_readiness, '_log'):
            if optimized:
                output = ml_readiness.load(now)
            else:
                # dev@77799c2: both reads were inside the same retry block.
                for _ in range(3):
                    try:
                        baseline_rows = read()
                        baseline_scans = index()
                        output = {'rows': baseline_rows, 'scans': baseline_scans}
                        break
                    except RuntimeError:
                        pass
        outputs.append(output)
        results.append(metrics.snapshot())
    return {'kind': 'synthetic_payload_estimate', 'fixture_rows': row_count,
            'scenario': 'first scan-index read fails after successful observation download',
            'outputs_equal': outputs[0] == outputs[1], 'before': results[0], 'after': results[1],
            'excluded': 'query text/parameters, TLS/protocol framing, provider accounting and timing comparison'}


def price_upload_model(symbols=8000, bars=250):
    import pandas as pd

    from db.prices import _serialize

    frame = pd.DataFrame({
        'Open': [10 + i / 100 for i in range(bars)],
        'High': [11 + i / 100 for i in range(bars)],
        'Low': [9 + i / 100 for i in range(bars)],
        'Close': [10.5 + i / 100 for i in range(bars)],
        'Volume': [100000] * bars,
    }, index=pd.date_range('2025-01-01', periods=bars))
    payload_bytes = len(_serialize(frame).encode())
    return {'kind': 'synthetic_serialized_payload_model', 'scenario': 'all requested frames are fresh Neon cache hits',
            'symbols': symbols, 'bars_per_symbol': bars, 'bytes_per_frame': payload_bytes,
            'before_client_to_db_payload_bytes': symbols * (payload_bytes + len('S00000')),
            'after_client_to_db_payload_bytes': 0,
            'before_upsert_rows': symbols, 'after_upsert_rows': 0,
            'db_to_client_savings_claimed': False,
            'excluded': 'SQL bytes, prepared/batched protocol, command acknowledgements and TLS'}


if __name__ == '__main__':
    print(json.dumps({'retry': benchmark(), 'price_upload': price_upload_model()}, indent=2))
