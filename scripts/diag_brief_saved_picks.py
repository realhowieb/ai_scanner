"""Compare saved Brief picks with model inference in separate fresh processes.

Read-only production-data diagnostic. Only aggregate timings, memory, module
presence and an output digest leave each child; no rows or tickers are logged.
No fitting, model replacement, notification or service restart is performed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def child(mode):
    from scheduler.morning_digest import _prebreakout_picks
    from scripts.diag_api_memory import rss_mb
    from ui.market_brief import _brief_scan_df

    frame = _brief_scan_df()
    if frame is None:
        raise RuntimeError('no saved scan available')
    before = rss_mb()
    started = time.monotonic()
    picks = _prebreakout_picks(frame, rescore=mode == 'rescore')
    elapsed = time.monotonic() - started
    gc.collect()
    result = {'mode': mode, 'run_id': frame.attrs.get('hsf_scan_run_id'),
              'saved_at': str(frame.attrs.get('hsf_scan_saved_at')), 'rows': len(frame),
              'pick_count': len(picks), 'seconds': round(elapsed, 3),
              'rss_before_mb': before, 'rss_after_mb': rss_mb(),
              'process_peak_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
              'xgboost_loaded': 'xgboost' in sys.modules,
              'pick_digest': hashlib.sha256(json.dumps([(p['symbol'], p['prob']) for p in picks]).encode()).hexdigest()}
    print('RESULT ' + json.dumps(result, default=str), flush=True)


def main():
    if len(sys.argv) > 1:
        child(sys.argv[1])
        return
    from scripts.diag_api_memory import _ensure_api_deps

    _ensure_api_deps()
    # Avoid startup workers and notifications; this is a standalone measurement.
    os.environ['RENDER'] = ''
    results = []
    for mode in ('saved', 'rescore'):
        run = subprocess.run([sys.executable, __file__, mode], cwd=ROOT, capture_output=True, text=True, timeout=300)
        lines = [line[7:] for line in run.stdout.splitlines() if line.startswith('RESULT ')]
        if run.returncode or not lines:
            raise RuntimeError(f'{mode} measurement failed (exit {run.returncode}); no private output printed')
        results.append(json.loads(lines[-1]))
    same_snapshot = all(r['run_id'] == results[0]['run_id'] and r['saved_at'] == results[0]['saved_at'] for r in results)
    print(json.dumps({'measurements': results, 'same_snapshot': same_snapshot,
                      'retained_rss_difference_mb': round(results[1]['rss_after_mb']-results[0]['rss_after_mb'], 1),
                      'same_picks': results[0]['pick_digest'] == results[1]['pick_digest'],
                      'note': 'Process RSS on the runner, not Render container usage. Saved values preserve prediction time; fresh rescoring can differ.'}, indent=2))


if __name__ == '__main__':
    main()
