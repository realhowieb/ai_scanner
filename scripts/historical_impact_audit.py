"""Read-only, aggregate-only replay of retained public snapshot performance.

No fitting, registry writes, schema setup, persisted performance or private payloads.
Current provider bars are a reconstruction, not the original provider archive.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from collections import Counter
from math import isfinite
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import market_calendar as cal
from analytics import track_method as method
from analytics.track_record import _bars_for, _ranked_symbols
from config import SCORE_EPOCH
from scripts.legacy_track_ranking import legacy_ranked_symbols

UTC = dt.timezone.utc


def legacy_select(rows, horizon, as_of):
    """Exact former UTC-date, age heuristic and newest-first daily selection."""
    seen, selected = set(), []
    for row in rows:
        stamp = row['signal_at']
        if stamp < as_of-dt.timedelta(days=45):
            break
        if stamp > as_of-dt.timedelta(days=int(horizon*1.6)+2):
            continue
        day = stamp.date()
        if day < SCORE_EPOCH or day.weekday() >= 5 or day in seen:
            continue
        seen.add(day)
        selected.append(row)
        if len(selected) >= 30:
            break
    return selected


def current_select(rows, as_of, *, raw_timestamps=False):
    seen, selected = set(), []
    for row in rows:
        if not row['public']:
            continue
        stamp = row['stored_created_at'] if raw_timestamps else row['signal_at']
        if not isinstance(stamp, dt.datetime) or stamp.tzinfo is None:
            continue
        day = stamp.astimezone(cal.ET).date()
        if stamp < as_of-dt.timedelta(days=45) or stamp > as_of or day < SCORE_EPOCH:
            continue
        if not cal.calendar_covered(day) or not cal.is_trading_day(day) or day in seen:
            continue
        seen.add(day)
        selected.append(row)
        if len(selected) >= 30:
            break
    return selected


def legacy_outcome(bars, day, horizon, mode='close'):
    """Frozen former price policy, with audit-only provenance added."""
    try:
        closes = bars['Close'].dropna()
        entry_pos = next((i for i, t in enumerate(closes.index) if t.date() >= day), None)
        if entry_pos is None or entry_pos+horizon >= len(closes):
            return None
        entry_day = closes.index[entry_pos].date()
        exit_day = closes.index[entry_pos+horizon].date()
        field, fallback = 'Close', False
        if mode == 'open' and 'Open' in bars.columns:
            entry = float(bars['Open'].reindex(closes.index).iloc[entry_pos])
            if entry != entry or entry <= 0:
                entry, fallback = float(closes.iloc[entry_pos]), True
            else:
                field = 'Open'
        else:
            entry = float(closes.iloc[entry_pos])
            fallback = mode == 'open'
        exit_ = float(closes.iloc[entry_pos+horizon])
        if entry <= 0:
            return None
        value = (exit_-entry)/entry
        if not isfinite(value):
            return None  # nonfinite old results are counted separately as unavailable
        entry_at = cal.session_bounds_utc(entry_day)[0 if field == 'Open' else 1]
        return {'return': value, 'entry_at': entry_at, 'exit_day': exit_day,
                'entry_day': entry_day, 'open_fallback': fallback}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


def metrics(rows):
    if not rows:
        return {'n': 0, 'avg_absolute_return': None, 'avg_excess_return': None,
                'median_excess_return': None, 'positive_return_rate': None, 'benchmark_outperformance_rate': None}
    absolute, excess = [r[0] for r in rows], [r[0]-r[1] for r in rows]
    return {'n': len(rows), 'avg_absolute_return': mean(absolute), 'avg_excess_return': mean(excess),
            'median_excess_return': median(excess), 'positive_return_rate': mean(v > 0 for v in absolute),
            'benchmark_outperformance_rate': mean(v > 0 for v in excess),
            'negative_stock_but_beat_spy': sum(a < 0 and a-b > 0 for a,b in rows),
            'positive_stock_but_underperformed_spy': sum(a > 0 and a-b < 0 for a,b in rows)}


def compare(selections, bars, horizon, as_of):
    old, paired_old, paired_new = [], [], []
    counters, excluded = Counter(), Counter()
    for stamp, picks in selections:
        for symbol in picks:
            counters['selected'] += 1
            stock_bars, spy_bars = _bars_for(bars, symbol), _bars_for(bars, 'SPY')
            a, b = legacy_outcome(stock_bars, stamp.date(), horizon), legacy_outcome(spy_bars, stamp.date(), horizon)
            spec = method.window(stamp, horizon)
            new_a, new_b = method.outcome(stock_bars, spec, now=as_of), method.outcome(spy_bars, spec, now=as_of)
            if a and b:
                old.append((a['return'], b['return']))
                counters['legacy_entry_before_snapshot'] += a['entry_at'] < stamp
                counters['legacy_stock_spy_window_mismatch'] += (a['entry_day'],a['exit_day']) != (b['entry_day'],b['exit_day'])
            open_a, open_b = legacy_outcome(stock_bars, stamp.date(), horizon, 'open'), legacy_outcome(spy_bars, stamp.date(), horizon, 'open')
            counters['legacy_open_fallback_picks'] += bool(open_a and open_b and (open_a['open_fallback'] or open_b['open_fallback']))
            reason = new_b.get('reason') or new_a.get('reason')
            if reason:
                excluded[reason] += 1
                continue
            if a and b:
                paired_old.append((a['return'],b['return']))
                paired_new.append((new_a['return'],new_b['return']))
                counters['same_exit_session_paired'] += a['exit_day'] == spec['exit_day'] == b['exit_day']
    pold, pnew = metrics(paired_old), metrics(paired_new)
    return {'counts': dict(counters), 'legacy_reconstruction': metrics(old),
            'paired_legacy': pold, 'paired_corrected': pnew,
            'paired_excess_delta_pp': 100*(pnew['avg_excess_return']-pold['avg_excess_return']) if paired_old else None,
            'corrected_exclusions': dict(excluded)}


def main():
    import pandas as pd

    from db import engine

    conn = engine.get_neon_conn()
    if conn is None:
        raise RuntimeError('database unavailable')
    try:
        conn.execute('SET TRANSACTION READ ONLY')
        conn.execute('SET LOCAL statement_timeout = 60000')
        cur = conn.cursor()
        def query(sql, params=()):
            cur.execute(sql, params)
            rows = cur.fetchall()
            names = [d.name for d in cur.description]
            return [dict(r) if isinstance(r, dict) else dict(zip(names,r)) for r in rows]
        readonly = query("SELECT current_setting('transaction_read_only') AS read_only")[0]['read_only']
        if readonly != 'on':
            raise RuntimeError('read-only guard unavailable')
        saved = query("SELECT DISTINCT ON(horizon_days,ranking) horizon_days,ranking,avg_return,median_return,win_rate,"
                      "sample_size,runs_used,computed_at FROM signal_track_record WHERE horizon_days IN(1,5,20) "
                      "AND ranking IN('breakout','prebreakout_snapshot_v1') ORDER BY horizon_days,ranking,computed_at DESC")
        now = dt.datetime.now(UTC)
        earliest = min(r['computed_at'] for r in saved)-dt.timedelta(days=46)
        source = query("SELECT CASE WHEN username='cron' AND label='daily_snapshot' THEN id END AS id, "
                       "created_at AS stored_created_at, created_at AT TIME ZONE 'UTC' AS signal_at, "
                       "coalesce(username='cron' AND label='daily_snapshot',false) AS public, "
                       "CASE WHEN username='cron' AND label='daily_snapshot' THEN "
                       "(SELECT jsonb_agg(jsonb_build_object('Ticker',coalesce(x->'Ticker',x->'Symbol'),"
                       "'BreakoutScore',x->'BreakoutScore','PreBreakoutProb%',x->'PreBreakoutProb%',"
                       "'PreBreakoutProb',x->'PreBreakoutProb','ModelProvenance',x->'ModelProvenance') ORDER BY n) "
                       "FROM jsonb_array_elements(results_json::jsonb) WITH ORDINALITY a(x,n)) END AS rows "
                       "FROM runs WHERE is_snapshot AND created_at >= %s AND created_at <= %s "
                       "ORDER BY created_at DESC,id DESC LIMIT 800", (earliest.replace(tzinfo=None),now.replace(tzinfo=None)))
        if len(source) >= 800:
            raise RuntimeError('snapshot cap reached; narrow audit before continuing')
        for row in source:
            row['frame'] = pd.DataFrame(row['rows'] or [])
        cohorts, all_symbols = [], {'SPY'}
        for stored in saved:
            ranking = 'prebreakout' if stored['ranking']=='prebreakout_snapshot_v1' else 'breakout'
            old_cohort = legacy_select(source, stored['horizon_days'],stored['computed_at'])
            private = sum(not r['public'] for r in old_cohort)
            selections = [(r['signal_at'],legacy_ranked_symbols(r['frame'],ranking,5)) for r in old_cohort if r['public']]
            for _, picks in selections:
                all_symbols.update(picks)
            cohorts.append((stored, ranking, selections, len(old_cohort), private))
        current = current_select(source,now)
        new_selections = {rk: [(r['signal_at'],_ranked_symbols(r['frame'],rk,5)) for r in current] for rk in ('breakout','prebreakout')}
        for rows in new_selections.values():
            for _, picks in rows:
                all_symbols.update(picks)
        if len(all_symbols)>600:
            raise RuntimeError('symbol cap exceeded')
        # The connection is closed before provider requests; no writer or schema helper is called.
        cur.close()
    finally:
        conn.close()
    from data.price_alpaca import download_multi_alpaca, get_alpaca_data_feed
    logging.getLogger('data.price_alpaca').setLevel(logging.CRITICAL)
    bars = download_multi_alpaca(sorted(all_symbols),period='90d',interval='1d',prepost=False,timeout_s=25.0)
    results = []
    for stored, ranking, selections, total, private in cohorts:
        result = compare(selections,bars,stored['horizon_days'],stored['computed_at'])
        rebuilt = result['legacy_reconstruction']
        result.update({'ranking':ranking,'horizon':stored['horizon_days'],'stored':stored,
                       'old_cohort_snapshots':total,'nonpublic_snapshots_not_exported_or_replayed':private,
                       'snapshots_without_usable_saved_ranking':sum(not picks for _,picks in selections),
                       'old_ui_rate_gate_passed':stored['sample_size']>=25,
                       'reconstruction_minus_stored_excess_pp':100*(rebuilt['avg_excess_return']-stored['avg_return']) if rebuilt['n'] else None,
                       'reconstruction_n_matches_stored':rebuilt['n']==stored['sample_size']})
        results.append(result)
    full = []
    for h in (1,5,20):
        for ranking, selections in new_selections.items():
            result = method.evaluate(selections,bars,h,now=now)
            full.append({'ranking':ranking,**{k:v for k,v in result.items() if k not in ('observations','daily')},
                         'ranking_unavailable_snapshots':sum(not picks for _,picks in selections)})
    report = {'audit_as_of':now,'source_revision':'ff922531c2eb6ae70583e2dbab174291534e0075',
              'database_read_only':readonly,'provider':'Alpaca','feed':get_alpaca_data_feed(),
              'bar_adjustment':'raw','bar_source':'current provider reconstruction, not original archived bars',
              'requested_public_symbols':len(all_symbols),'returned_symbols':len(bars),
              'public_daily_snapshots_after_utc_normalization':len(current),
              'public_daily_snapshots_with_deployed_naive_reader':len(current_select(source,now,raw_timestamps=True)),
              'naive_stored_timestamp_rows':sum(r['stored_created_at'].tzinfo is None for r in source),
              'stored_comparisons':results,'corrected_current_cohort':full,
              'limits':['Private/unknown-source snapshot payloads are never exported or replayed.',
                        'Original bar versions and original selected memberships were not archived; exact viewing history is unknown.',
                        'Paired comparisons change timing/horizon policy on the same picks; current-cohort totals also change selection and coverage.',
                        'This quantifies ranking-study reconstruction, not production model predictive accuracy.']}
    print('IMPACT_REPORT_BEGIN')
    print(json.dumps(report,indent=2,default=str,allow_nan=False))
    print('IMPACT_REPORT_END')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
