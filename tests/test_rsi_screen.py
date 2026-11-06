"""RSI correctness and custom-scan screening contracts."""
import json

import pandas as pd
import pytest

from scan.rsi_screen import daily_rsi14, rsi_screen


def bars(up=0.5):
    closes = [100 + i * .5 for i in range(20)] + [109 - i for i in range(15)] + [95 + up * i for i in range(1, 5)]
    return pd.DataFrame({"Open": closes, "Close": closes, "High": [x + .1 for x in closes],
                         "Low": [x - .1 for x in closes], "Volume": [1000] * (len(closes)-1) + [1200]},
                        index=pd.date_range("2026-01-01", periods=len(closes)))


def test_wilder_reference_and_flat_or_one_direction_markets():
    prices = pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28])
    assert daily_rsi14(prices).iloc[-1] == pytest.approx(70.464135, abs=.000001)
    for prices, expected in [(range(1, 30), 100), (range(30, 1, -1), 0), ([10]*30, 50)]:
        assert daily_rsi14(pd.Series(prices)).iloc[-1] == expected


def test_inclusive_bounds_and_missing_history_fail_closed():
    frame = bars()
    value, _, _ = rsi_screen(frame)
    assert rsi_screen(frame, min_rsi=value, max_rsi=value)[2]
    assert not rsi_screen(frame, max_rsi=value-.01)[2]
    for frame in (frame.iloc[:10], frame.assign(Close=float("nan"))):
        assert rsi_screen(frame) == (None, False, True)
        assert not rsi_screen(frame, min_rsi=0, max_rsi=100)[2]


def test_rebound_requires_recovery_volume_and_up_close():
    assert rsi_screen(bars(), rebound=True, relative_volume=1)[1:]==(True, True)
    for frame, volume in [(bars(.3), 1.2), (bars(1.5), 1.2), (bars(), .99), (bars(), None)]:
        assert not rsi_screen(frame, rebound=True, relative_volume=volume)[2]
    frame = bars()
    frame.loc[frame.index[-1], "Close"] = frame.Close.iloc[-2] - .1
    assert not rsi_screen(frame, rebound=True, relative_volume=2)[2]
    assert daily_rsi14(bars().Close).iloc[:-1].equals(daily_rsi14(bars().Close.iloc[:-1]))


def test_engine_filters_before_top_n():
    from scan.breakout import run_breakout_scan
    strong = bars().assign(Close=list(range(100, 139)), Open=list(range(100, 139)),
                           High=list(range(101, 140)), Low=list(range(99, 138)))
    result = run_breakout_scan(price_data={"STRONG": strong, "RECOVER": bars()}, spy_df=None,
                              premarket=False, afterhours=False, unusual_volume=False, min_gap=0, min_price=1, max_price=1000,
                              max_rsi=40, top_n=1, diagnostics=False)
    assert list(result.Ticker) == ["RECOVER"]
    assert result.RSI14.iloc[0] < 40


def test_plan_preserves_bounds_and_rejects_conflicting_or_invalid_settings():
    from tests.test_api_scans import _plan
    plan = _plan({"universe": "sp500", "filters": {"min_rsi": 0, "max_rsi": 30}}, "premium")
    assert (plan["min_rsi"], plan["max_rsi"], plan["min_gap"]) == (0, 30, 0)
    for filters in ({"min_rsi": 40, "max_rsi": 30}, {"max_rsi": 101}, {"min_rsi": float("nan")}):
        with pytest.raises(ValueError):
            _plan({"universe": "sp500", "filters": filters}, "premium")
    with pytest.raises(ValueError, match="Score all"):
        _plan({"universe": "ticker", "ticker": "AAPL", "score_all": True, "filters": {"rebound": True}}, "premium")


def test_ai_builder_roundtrips_rsi_and_disables_score_all():
    from api.ai_premium import parse_builder
    result = parse_builder(json.dumps({"universe": "ticker", "ticker": "AAPL", "min_rsi": 28,
                                       "max_rsi": 32, "rebound": True, "score_all": True}), [])
    assert result["view"]["min_rsi"] == 28
    assert result["view"]["max_rsi"] == 32
    assert result["view"]["rebound"] is True
    assert result["view"]["score_all"] is False
