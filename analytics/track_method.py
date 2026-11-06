"""Versioned signal research: aligned daily-bar proxies, never simulated fills."""
from __future__ import annotations

import datetime as dt
from collections import Counter
from math import isfinite
from statistics import mean, median

from analytics import market_calendar as cal

METHOD = 'track-v2-next-open'


def window(signal_at: dt.datetime, horizon: int, mode='open'):
    """First session open/close strictly after saved availability; h sessions inclusive."""
    if not isinstance(signal_at, dt.datetime) or signal_at.tzinfo is None:
        return {'reason': 'SIGNAL_TIMESTAMP_UNAVAILABLE'}
    if mode not in ('open', 'close') or horizon < 1:
        raise ValueError('entry mode must be open/close and horizon positive')
    signal_at = signal_at.astimezone(dt.timezone.utc)
    day = signal_at.astimezone(cal.ET).date()
    for _ in range(370):
        if not cal.calendar_covered(day):
            return {'reason': 'CALENDAR_UNSUPPORTED'}
        if cal.is_trading_day(day):
            bounds = cal.session_bounds_utc(day)
            entry = bounds[0 if mode == 'open' else 1]
            if entry > signal_at:
                break
        day += dt.timedelta(days=1)
    entry_day = day
    sessions = 1 if mode == 'open' else 0
    while sessions < horizon:
        day += dt.timedelta(days=1)
        if not cal.calendar_covered(day):
            return {'reason': 'CALENDAR_UNSUPPORTED'}
        if cal.is_trading_day(day):
            sessions += 1
    return {'signal_at': signal_at.isoformat(), 'entry_at': entry.isoformat(),
            'exit_at': cal.session_bounds_utc(day)[1].isoformat(), 'entry_day': entry_day,
            'exit_day': day, 'entry_field': 'Open' if mode == 'open' else 'Close',
            'entry_policy': f'first_session_{mode}_strictly_after_snapshot',
            'methodology': METHOD if mode == 'open' else 'track-v2-next-close'}


def bar_date(value):
    # Alpaca daily timestamps are session labels (midnight ET). Naive/UTC-midnight
    # fixtures also represent date labels; converting UTC midnight to ET loses a day.
    return value.date() if hasattr(value, 'date') else value


def outcome(bars, spec, *, now=None):
    """Resolve exact common session dates. Never shift one symbol's window."""
    if spec.get('reason'):
        return {'reason': spec['reason']}
    now = now or dt.datetime.now(dt.timezone.utc)
    if dt.datetime.fromisoformat(spec['exit_at']) > now:
        return {'reason': 'INCOMPLETE_FORWARD_WINDOW'}
    if bars is None or getattr(bars, 'empty', True):
        return {'reason': 'MISSING_SYMBOL_BARS'}
    field = spec['entry_field']
    if field not in bars.columns:
        return {'reason': 'ENTRY_FIELD_UNAVAILABLE'}
    if 'Close' not in bars.columns:
        return {'reason': 'EXIT_FIELD_UNAVAILABLE'}
    dates = [bar_date(v) for v in bars.index]
    if any(day != day or not isinstance(day, dt.date) for day in dates) or len(dates) != len(set(dates)) or dates != sorted(dates):
        return {'reason': 'INVALID_BAR_INDEX'}
    if spec['entry_day'] not in dates or spec['exit_day'] not in dates:
        return {'reason': 'INCOMPLETE_FORWARD_WINDOW'}
    required = list(cal.trading_days_between(spec['entry_day'], spec['exit_day']))
    if any(day not in dates for day in required):
        return {'reason': 'INCOMPLETE_FORWARD_WINDOW'}
    def price(day, column):
        try:
            value = float(bars.iloc[dates.index(day)][column])
            return value if isfinite(value) and value > 0 else None
        except (TypeError, ValueError, OverflowError):
            return None
    entry, exit_ = price(spec['entry_day'], field), price(spec['exit_day'], 'Close')
    if entry is None:
        return {'reason': 'INVALID_ENTRY_PRICE'}
    if exit_ is None:
        return {'reason': 'INVALID_EXIT_PRICE'}
    value = (exit_ - entry) / entry
    if not isfinite(value):
        return {'reason': 'INVALID_RETURN'}
    return {'return': value, 'entry_price': entry, 'exit_price': exit_}


def evaluate(selections, bars_by_symbol, horizon, *, mode='open', benchmark='SPY', now=None):
    """Selections are (aware snapshot timestamp, ranked symbol list). One row/pick."""
    from analytics.track_record import _bars_for

    evidence, absolute, excess, daily = [], [], [], []
    selected_symbols, evaluated_symbols, signal_dates = set(), set(), set()
    excluded = Counter()
    for selection in selections:
        signal, symbols = selection[:2]
        snapshot_id = selection[2] if len(selection) > 2 else None
        spec = window(signal, horizon, mode)
        day = signal.astimezone(cal.ET).date() if isinstance(signal, dt.datetime) and signal.tzinfo else None
        if symbols and day:
            signal_dates.add(day)
        spy_bars = _bars_for(bars_by_symbol, benchmark)
        spy = outcome(spy_bars, spec, now=now)
        vals = []
        for symbol in symbols:
            selected_symbols.add(symbol)
            record = {'symbol': symbol, 'snapshot_id': snapshot_id, **{k: v for k, v in spec.items() if k not in ('entry_day', 'exit_day')},
                      'signal_at': signal.isoformat() if hasattr(signal, 'isoformat') else None}
            stock = outcome(_bars_for(bars_by_symbol, symbol), spec, now=now)
            if spec.get('reason'):
                reason = spec['reason']
            elif spy_bars is None or getattr(spy_bars, 'empty', True):
                reason = 'MISSING_BENCHMARK_BARS'
            elif spy.get('reason'):
                reason = 'BENCHMARK_' + spy['reason']
            else:
                reason = stock.get('reason')
            if reason:
                excluded[reason] += 1
                evidence.append({**record, 'exclusion_reason': reason})
                continue
            a, b = stock['return'], spy['return']
            absolute.append(a)
            excess.append(a-b)
            vals.append(a-b)
            evaluated_symbols.add(symbol)
            evidence.append({**record, 'absolute_return': a, 'benchmark_return': b, 'excess_return': a-b,
                             'entry_price': stock['entry_price'], 'exit_price': stock['exit_price'],
                             'benchmark_entry_price': spy['entry_price'], 'benchmark_exit_price': spy['exit_price']})
        if vals:
            daily.append((day, mean(vals), len(vals)))
    n, selected = len(absolute), len(evidence)
    avg, med = (mean(excess), median(excess)) if n else (None, None)
    beat = sum(v > 0 for v in excess)/n if n else None
    return {'methodology': METHOD if mode == 'open' else 'track-v2-next-close',
            'entry_policy': f'first_session_{mode}_strictly_after_snapshot', 'horizon_days': horizon,
            'avg_absolute_return': mean(absolute) if n else None, 'median_absolute_return': median(absolute) if n else None,
            'avg_excess_return': avg, 'median_excess_return': med,
            'positive_return_rate': sum(v > 0 for v in absolute)/n if n else None,
            'benchmark_outperformance_rate': beat,
            # Deprecated aliases keep their original benchmark-relative meaning.
            'avg_return': avg, 'median_return': med, 'win_rate': beat,
            'sample_size': n, 'observation_count': n, 'selected_count': selected, 'evaluated_count': n,
            'excluded_count': selected-n, 'coverage_pct': 100*n/selected if selected else None,
            'exclusions': dict(sorted(excluded.items())), 'unique_ticker_count': len(evaluated_symbols),
            'selected_unique_ticker_count': len(selected_symbols), 'signal_date_count': len(signal_dates),
            'runs_used': len(daily), 'benchmark': benchmark, 'daily': daily, 'observations': evidence}
