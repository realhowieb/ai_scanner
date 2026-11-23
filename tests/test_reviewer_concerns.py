"""Market condition rule, public track record, recap last-week results, execution-aware
trade plan and the live breakout-trigger alert."""
from __future__ import annotations

import datetime as dt
import math

from analytics import market_condition as mc
from api import public_record as pr
from billing_service import realtime_alerts as ra
from ui.trade_plan import build_trade_plan


# ---- market condition -------------------------------------------------------------------
def _trend(n=80, start=100.0, step=0.003):
    return [start * (1 + step) ** i for i in range(n)]


def test_steady_uptrend_is_trending_up():
    c = mc.classify(_trend())
    assert c["condition"] == mc.TRENDING_UP and c["label"] == "Trending up"
    assert c["spy_vs_sma20_pct"] > 0 and c["sma20_rising"]
    assert "20-day average" in mc.why(c)


def test_downtrend_is_risk_off():
    assert mc.classify(_trend(step=-0.003))["condition"] == mc.RISK_OFF


def test_big_swings_are_risk_off_even_in_an_uptrend():
    closes = [100 * (1.0 + 0.002 * i) * (1.03 if i % 2 else 0.97) for i in range(80)]
    c = mc.classify(closes)
    assert c["volatility_20d_pct"] >= mc.HIGH_VOL and c["condition"] == mc.RISK_OFF


def test_mixed_is_choppy():
    up = _trend(60)
    closes = up + [up[-1] * (1 - 0.004 * i) for i in range(1, 6)]   # dips under the 20-day average
    assert mc.classify(closes)["condition"] == mc.CHOPPY


def test_too_few_closes_is_none():
    assert mc.classify(_trend(30)) is None


def test_by_day_is_point_in_time_and_lookup_uses_the_close_before():
    series = [((dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat(), v) for i, v in enumerate(_trend(90))]
    labels = mc.by_day(series)
    assert min(labels) == series[mc.MIN_CLOSES - 1][0]
    # Changing a later close can't change an earlier label.
    later = series[:70] + [(d, v * 0.5) for d, v in series[70:]]
    assert {d: c for d, c in mc.by_day(later).items() if d < series[70][0]} == \
           {d: c for d, c in labels.items() if d < series[70][0]}
    day = series[60][0]
    assert mc.condition_on(day, labels) == labels[series[59][0]]
    assert mc.condition_on(series[0][0], labels) is None


# ---- public track record ----------------------------------------------------------------
UTC = dt.timezone.utc


def _row(t, fired, score, ret=None, bench=None, computed=True):
    return {"ticker": t, "fired_at": fired, "score": score, "setup": "breakout", "return_5d": ret,
            "benchmark_5d": bench, "computed_at": fired + dt.timedelta(days=8) if computed else None}


def test_daily_top_takes_first_run_top_n_and_marks_status():
    d1 = dt.datetime(2026, 11, 2, 13, 35, tzinfo=UTC)   # 8:35 ET
    later = d1 + dt.timedelta(hours=3)
    rows = [_row(f"T{i}", d1, 90 - i, 0.01 * i, 0.005) for i in range(12)]
    rows += [_row("LATE", later, 99, 0.5, 0.0), _row("LOW", d1, 39, 0.1, 0.0)]
    rows += [_row("PEND", d1 + dt.timedelta(days=1), 80, computed=False),
             _row("NOPX", d1 + dt.timedelta(days=1), 70, ret=None)]
    picks = pr.daily_top(rows)
    day1 = [p for p in picks if p["day"] == "2026-11-02"]
    assert [p["ticker"] for p in day1] == [f"T{i}" for i in range(10)]   # first run only, top 10, >= 40
    assert day1[3]["excess_5d"] == 0.03 - 0.005
    status = {p["ticker"]: p["status"] for p in picks}
    assert status["PEND"] == "pending" and status["NOPX"] == "unavailable"


def test_verdict_needs_days_and_a_clear_interval():
    base = dt.datetime(2026, 9, 14, 13, 35, tzinfo=UTC)
    rows = []
    for d in range(30):
        for i in range(5):
            rows.append(_row(f"S{d}_{i}", base + dt.timedelta(days=d), 60, 0.03 + 0.001 * i, 0.0))
    out = pr.build(rows)
    assert out["verdict"] == pr.AHEAD and out["summary"]["days"] == 30
    assert out["recent"][0]["day"] >= out["recent"][-1]["day"]
    # Same picks over too few days: no verdict either way.
    few = [r for r in rows if r["fired_at"] < base + dt.timedelta(days=5)]
    assert pr.build(few)["verdict"] == pr.NO_EDGE
    # Mixed results straddling zero: no edge.
    mixed = [{**r, "return_5d": (0.02 if i % 2 else -0.02)} for i, r in enumerate(rows)]
    m = pr.build(mixed)
    assert m["verdict"] == pr.NO_EDGE and "proven edge" in m["verdict_text"]


def test_build_groups_by_condition():
    base = dt.datetime(2026, 10, 5, 13, 35, tzinfo=UTC)
    rows = [_row("A", base, 70, 0.02, 0.01), _row("B", base + dt.timedelta(days=1), 70, -0.01, 0.0)]
    out = pr.build(rows, lambda day: mc.CHOPPY if day == "2026-10-05" else mc.RISK_OFF)
    groups = {g["condition"]: g for g in out["by_condition"]}
    assert groups["choppy"]["picks"] == 1 and math.isclose(groups["choppy"]["avg_excess"], 0.01)
    assert groups["risk_off"]["beat_rate"] == 0.0 and groups["trending_up"]["picks"] == 0
    assert out["recent"][0]["condition"] == "risk_off"


def test_week_outcomes_best_and_worst(monkeypatch):
    monday = dt.date(2026, 11, 2)
    fired = dt.datetime(2026, 11, 2, 13, 35, tzinfo=UTC)
    rows = [_row(t, fired, 80 - i, r, 0.01) for i, (t, r) in enumerate([("W", -0.05), ("X", 0.0), ("Y", 0.04), ("Z", 0.10)])]
    monkeypatch.setattr(pr, "load_rows", lambda start=None, end=None: rows)
    out = pr.week_outcomes(monday)
    assert out["picks"] == 4 and out["best"][0]["ticker"] == "Z" and out["worst"][0]["ticker"] == "W"
    assert math.isclose(out["avg_excess"], (-0.05 + 0.0 + 0.04 + 0.10) / 4 - 0.01)
    monkeypatch.setattr(pr, "load_rows", lambda start=None, end=None: [])
    assert pr.week_outcomes(monday) is None


# ---- execution-aware trade plan ---------------------------------------------------------
ROW = {"Ticker": "ABC", "Last": 50.0, "Volatility20D%": 8.0, "DollarVol20": 50_000_000.0, "BreakoutPos20D": 0.98}


def test_plan_without_quote_matches_the_scan_plan():
    p = build_trade_plan(ROW)
    assert p["entry_source"] == "scan" and p["entry"] == 50.0 and p["stop_pct"] == 4.0
    assert p["trigger"] == round(50 / 0.98, 2) and p["chase_limit"] == 51.0 and p["warnings"] == []


def test_live_price_past_chase_line_warns_and_rebuilds_levels():
    p = build_trade_plan(ROW, live_price=52.0, bid=51.98, ask=52.02, live_at="2026-11-19T15:00:00Z")
    assert p["entry_source"] == "live" and p["entry"] == 52.0 and p["moved_since_scan_pct"] == 4.0
    assert p["stop"] == round(52 * 0.96, 2) and any("don't-chase" in w for w in p["warnings"])


def test_live_price_below_scan_stop_says_setup_failed():
    p = build_trade_plan(ROW, live_price=47.0)
    assert any("already failed" in w for w in p["warnings"])


def test_thin_stock_wide_spread_and_big_position_warn():
    row = {**ROW, "DollarVol20": 2_000_000.0}
    p = build_trade_plan(row, account_size=1_000_000, risk_pct=2, live_price=50.0, bid=49.7, ask=50.3)
    text = " ".join(p["warnings"])
    assert "Thinly traded" in text and "Wide spread" in text and "dollar volume" in text
    assert p["liquidity"]["spread_pct"] > 1 and p["liquidity"]["position_pct_of_volume"] > 1


# ---- breakout trigger alert -------------------------------------------------------------
def _at(h, m):
    return dt.datetime(2026, 11, 19, h, m, tzinfo=dt.timezone(dt.timedelta(hours=-5))).astimezone(UTC)


def test_volume_pace():
    assert ra.volume_pace(1_000_000, 1_000_000, _at(16, 0)) == 1.0
    assert math.isclose(ra.volume_pace(500_000, 1_000_000, _at(12, 45)), 1.0)
    assert ra.volume_pace(500_000, 1_000_000, _at(9, 32)) is None    # opening minutes
    assert ra.volume_pace(500_000, 1_000_000, _at(17, 0)) is None    # after the close
    assert ra.volume_pace(None, 1_000_000, _at(12, 0)) is None


def test_trigger_alert_needs_price_and_volume_pace(monkeypatch):
    monkeypatch.setattr(ra, "_avg_volume", lambda t: 1_000_000)
    alert = {"ticker": "ABC", "threshold": 51.0, "direction": None, "alert_type": "trigger"}
    snap = {"last": 51.2, "volume": 900_000, "trade_day": "2026-11-19", "daily_day": "2026-11-19"}
    msg = ra.evaluate_alert(alert, snap, today="2026-11-19", now_utc=_at(12, 45))
    assert msg and "broke above your 51.00 trigger" in msg and "1.8x" in msg
    assert ra.evaluate_alert(alert, {**snap, "volume": 400_000}, today="2026-11-19", now_utc=_at(12, 45)) is None
    assert ra.evaluate_alert(alert, {**snap, "last": 50.9}, today="2026-11-19", now_utc=_at(12, 45)) is None


def test_trigger_is_a_creatable_alert_type():
    from api import user_data
    from db.alerts import ALERT_TYPES

    assert "trigger" in ALERT_TYPES and "trigger" in user_data.ALERT_RULES
    spec = next(t for t in user_data.alert_types() if t["type"] == "trigger")
    assert spec["needs_ticker"] and spec["threshold"]["min_exclusive"]


def test_prior_volumes_keeps_newest_sessions_and_drops_today():
    days = [f"2026-11-{d:02d}" for d in range(1, 24)]
    bars = [{"t": f"{d}T05:00:00Z", "v": i + 1} for i, d in enumerate(days)]
    # During the session: today's partial bar is left out, the 20 before it kept.
    assert ra._prior_volumes(bars, "2026-11-23") == [float(v) for v in range(3, 23)]
    # Pre-market (no bar for today yet): the newest session counts.
    assert ra._prior_volumes(bars[:-1], "2026-11-23") == [float(v) for v in range(3, 23)]
