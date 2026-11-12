"""Premium auto-score: the scheduled scan scores every earnings reporter once per slot."""
import datetime as dt
from unittest import mock

from scheduler import earnings_autoscore as ea
from scheduler import earnings_email as ee

NOW = dt.datetime(2026, 10, 27, 14, 35, tzinfo=dt.timezone.utc)   # 10:35 ET Tuesday
TUE = dt.date(2026, 10, 27)


def _cal(n=3):
    return [{"ticker": f"T{i}", "date": TUE + dt.timedelta(days=i % 8), "time": "bmo"} for i in range(n)]


def test_window_tickers_soonest_first_and_capped():
    rows = [{"ticker": "LATE", "date": TUE + dt.timedelta(days=6), "time": None},
            {"ticker": "SOON", "date": TUE, "time": "amc"},
            {"ticker": "MID", "date": TUE + dt.timedelta(days=2), "time": "bmo"}]
    with mock.patch.object(ee, "reporters", return_value=rows) as rep:
        assert ea.window_tickers(TUE, TUE + dt.timedelta(days=7)) == ["SOON", "MID", "LATE"]
        assert ea.window_tickers(TUE, TUE + dt.timedelta(days=7), limit=2) == ["SOON", "MID"]
    assert rep.call_args.args == (TUE, TUE + dt.timedelta(days=7))


def test_window_tickers_puts_tonight_and_next_day_before_a_busy_today():
    # Thursday: a busy today must not crowd out Friday's reporters (Nov 12 2026: 517 today).
    thu, fri, mon = dt.date(2026, 11, 12), dt.date(2026, 11, 13), dt.date(2026, 11, 16)
    rows = [{"ticker": "AMOR", "date": thu, "time": "bmo"},
            {"ticker": "ANON", "date": thu, "time": None},
            {"ticker": "ZTON", "date": thu, "time": "amc"},
            {"ticker": "ZFRI", "date": fri, "time": None},
            {"ticker": "BFRI", "date": fri, "time": "bmo"},
            {"ticker": "AMON", "date": mon, "time": "bmo"}]
    after_open = dt.datetime(2026, 11, 12, 21, 0, tzinfo=dt.timezone.utc)    # 4 pm ET
    before_open = dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.timezone.utc)   # 7 am ET
    with mock.patch.object(ee, "reporters", return_value=rows):
        assert ea.window_tickers(thu, thu + dt.timedelta(days=7), now=after_open) == \
            ["ZTON", "BFRI", "ZFRI", "ANON", "AMON", "AMOR"]
        assert ea.window_tickers(thu, thu + dt.timedelta(days=7), limit=3, now=after_open) == \
            ["ZTON", "BFRI", "ZFRI"]
        assert ea.window_tickers(thu, thu + dt.timedelta(days=7), now=before_open)[:4] == \
            ["AMOR", "ZTON", "BFRI", "ZFRI"]


def test_run_scores_the_week_and_saves_one_pass():
    setups = [{"ticker": "T0", "score": 70}]
    with mock.patch.object(ee, "reporters", return_value=_cal()), \
            mock.patch.object(ea, "scan_setups", return_value=setups) as scan, \
            mock.patch("db.earnings_autoscore.save", return_value=12) as save:
        assert ea.run_earnings_autoscore("premarket", now=NOW) == 12
    # 10:35 ET: this morning's T0 already reported, so it goes last.
    assert scan.call_args.args == (["T1", "T2", "T0"],) and scan.call_args.kwargs == {"premarket": True}
    kw = save.call_args.kwargs
    assert save.call_args.args == (setups,)
    assert (kw["session"], kw["window_start"], kw["window_end"], kw["symbols"]) == \
        ("premarket", TUE, TUE + dt.timedelta(days=7), 3)


def test_run_is_best_effort_and_can_be_switched_off(monkeypatch):
    with mock.patch.object(ee, "reporters", return_value=_cal()), \
            mock.patch.object(ea, "scan_setups", side_effect=RuntimeError("feed")), \
            mock.patch("db.earnings_autoscore.save") as save:
        assert ea.run_earnings_autoscore("regular", now=NOW) is None
    save.assert_not_called()
    with mock.patch.object(ee, "reporters", return_value=[]), mock.patch.object(ea, "scan_setups") as scan:
        assert ea.run_earnings_autoscore("regular", now=NOW) is None
    scan.assert_not_called()
    monkeypatch.setenv("HSF_EARNINGS_AUTOSCORE", "0")
    with mock.patch.object(ee, "reporters") as rep:
        assert ea.run_earnings_autoscore("regular", now=NOW) is None
    rep.assert_not_called()


def test_fresh_scores_match_the_session_kind():
    stamp = {"id": 3, "session": "regular"}
    rows = [{"ticker": "BRK.B", "score": 50}, {"ticker": "POOL", "score": 80}]
    with mock.patch("db.earnings_autoscore.latest_stamp", return_value=stamp), \
            mock.patch("db.earnings_autoscore.load_setups", return_value=rows):
        assert set(ea.fresh_scores()) == {"BRKB", "POOL"}
        assert ea.fresh_scores(premarket=True) is None
    with mock.patch("db.earnings_autoscore.latest_stamp", side_effect=RuntimeError("db down")):
        assert ea.fresh_scores() is None


def test_email_scoring_reuses_a_fresh_pass():
    fresh = {"POOL": {"ticker": "POOL", "score": 80, "last": 190.0, "chg_pct": 1.0}}
    with mock.patch.object(ea, "fresh_scores", return_value=fresh), mock.patch.object(ea, "scan_setups") as scan:
        assert ee.score_tickers(["POOL"]) == {"POOL": {"score": 80, "last": 190.0, "chg_pct": 1.0}}
    scan.assert_not_called()
    with mock.patch.object(ea, "fresh_scores", return_value=None), \
            mock.patch.object(ea, "scan_setups", return_value=[{"ticker": "POOL", "score": 5}]) as scan:
        assert ee.score_tickers(["POOL"])["POOL"]["score"] == 5
    assert scan.call_args.args == (["POOL"],)


def test_premium_only_flag():
    from domain.plans import FEATURE_MIN_TIER
    from ui.pricing import plan_highlights

    assert FEATURE_MIN_TIER["can_earnings_autoscore"] == "premium"
    assert any("earnings reporter" in h for h in plan_highlights()["premium"])
    assert not any("earnings reporter" in h for h in plan_highlights()["pro"])


def test_email_scores_names_the_pass_did_not_reach():
    fresh = {"POOL": {"ticker": "POOL", "score": 80, "last": 190.0, "chg_pct": 1.0}}
    with mock.patch.object(ea, "fresh_scores", return_value=fresh), \
            mock.patch.object(ea, "scan_setups", return_value=[{"ticker": "EIC", "score": 61}]) as scan:
        out = ee.score_tickers(["POOL", "EIC"])
    assert scan.call_args.args == (["EIC"],)
    assert out["POOL"]["score"] == 80 and out["EIC"]["score"] == 61
    # A failed top-up keeps the pass's scores instead of blanking everything.
    with mock.patch.object(ea, "fresh_scores", return_value=fresh), \
            mock.patch.object(ea, "scan_setups", side_effect=RuntimeError("feed")):
        assert set(ee.score_tickers(["POOL", "EIC"])) == {"POOL"}
