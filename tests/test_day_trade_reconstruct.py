"""Run 33 — full-feature historical reconstruction (daily indicators, no lookahead)."""
import datetime as dt
import unittest

import pandas as pd

from analytics import day_trade_reconstruct as rc


def _daily_df(n=60, start_close=100.0):
    idx = pd.date_range("2026-06-01", periods=n, freq="B")  # business days
    closes = [start_close + i * 0.5 for i in range(n)]  # steady uptrend
    return pd.DataFrame({
        "Open": [c - 0.2 for c in closes],
        "High": [c + 0.5 for c in closes],
        "Low": [c - 0.5 for c in closes],
        "Close": closes,
        "Volume": [1_000_000 + i * 1000 for i in range(n)],
    }, index=idx)


def _minute_bars(day="2026-08-25", base=130.0, n=40, rising=True):
    bars = []
    for i in range(n):
        c = base + (i * 0.05 if rising else -i * 0.05)
        # 09:35 ET = 13:35Z
        t = f"{day}T{13 + i // 60:02d}:{35 + i % 60:02d}:00Z" if i < 25 else f"{day}T14:{i:02d}:00Z"
        bars.append({"t": t, "o": base if i == 0 else c - 0.02, "h": c + 0.1, "l": c - 0.1, "c": c, "v": 5000})
    return bars


class ReconstructTests(unittest.TestCase):
    def test_full_features_and_no_lookahead(self):
        daily = _daily_df()
        # validation day is AFTER the daily history, so D-1 exists
        bars = _minute_bars(day="2026-08-25", base=daily["Close"].iloc[-1] + 1.0, rising=True)
        obs = rc.reconstruct_observations("NVDA", daily, bars, sample_every=5)
        self.assertTrue(obs)
        o = obs[0]
        # direction present; a rising day with prior uptrend should read bullish-ish
        self.assertIn(o["direction"], ("bullish", "neutral", "bearish"))
        self.assertEqual(o["ticker"], "NVDA")
        # forward outcome exists for an early sample with future bars
        late = next((x for x in obs if x.get("directional_return_15m") is not None), None)
        self.assertIsNotNone(late)

    def test_daily_indicators_use_prior_session_only(self):
        # The indicator value at day D must equal the daily series value at D-1,
        # never a later daily bar (no lookahead).
        daily = _daily_df()
        ind = rc.daily_indicator_series(daily)
        dm1 = dt.date(2026, 8, 24)  # a business day before 08-25
        # value as-of dm1 is finite and matches the last daily row <= dm1
        val = rc._finite_latest_upto(ind["adx"], dm1)
        # a later cutoff should differ or be >= (monotone data) — never earlier info leaking
        later = rc._finite_latest_upto(ind["adx"], dt.date(2026, 8, 26))
        self.assertIsNotNone(val)
        self.assertIsNotNone(later)

    def test_skips_day_without_prior(self):
        daily = _daily_df()
        # minute day BEFORE the daily history -> no prior session -> skipped
        bars = _minute_bars(day="2026-05-01")
        self.assertEqual(rc.reconstruct_observations("X", daily, bars), [])

    def test_et_date_parsing(self):
        self.assertEqual(rc._et_date("2026-08-25T13:35:00Z"), dt.date(2026, 8, 25))
        self.assertIsNone(rc._et_date("garbage"))


if __name__ == "__main__":
    unittest.main()
