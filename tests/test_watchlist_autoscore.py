"""Premium auto-score: the scheduled scan scores watchlist stocks the market scan didn't rank."""
from unittest import mock

from scheduler import watchlist_autoscore as wa


def test_tickers_skip_ranked_names_without_shrinking_the_cap():
    with mock.patch.object(wa, "ranked_tickers", return_value={"NVDA", "AMD"}), \
            mock.patch("db.watchlist_autoscore.watched_tickers", return_value=["NVDA", "KO", "AMD", "AAPL", "T"]) as w:
        assert wa.tickers_to_score(limit=2) == ["KO", "AAPL"]
    assert w.call_args.args == (4,)


def test_run_scores_and_saves_one_pass():
    setups = [{"ticker": "KO", "score": 40}]
    with mock.patch.object(wa, "tickers_to_score", return_value=["KO", "AAPL"]), \
            mock.patch("scheduler.earnings_autoscore.scan_setups", return_value=setups) as scan, \
            mock.patch("db.watchlist_autoscore.save", return_value=5) as save:
        assert wa.run_watchlist_autoscore("premarket") == 5
    assert scan.call_args.args == (["KO", "AAPL"],) and scan.call_args.kwargs == {"premarket": True}
    assert save.call_args.args == (setups,)
    assert (save.call_args.kwargs["session"], save.call_args.kwargs["symbols"]) == ("premarket", 2)


def test_run_is_best_effort_and_can_be_switched_off(monkeypatch):
    with mock.patch.object(wa, "tickers_to_score", return_value=["KO"]), \
            mock.patch("scheduler.earnings_autoscore.scan_setups", side_effect=RuntimeError("feed")), \
            mock.patch("db.watchlist_autoscore.save") as save:
        assert wa.run_watchlist_autoscore() is None
    save.assert_not_called()
    with mock.patch.object(wa, "tickers_to_score", return_value=[]), \
            mock.patch("scheduler.earnings_autoscore.scan_setups") as scan:
        assert wa.run_watchlist_autoscore() is None
    scan.assert_not_called()
    monkeypatch.setenv("HSF_WATCHLIST_AUTOSCORE", "0")
    with mock.patch.object(wa, "tickers_to_score") as t:
        assert wa.run_watchlist_autoscore() is None
    t.assert_not_called()
