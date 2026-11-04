"""Deterministic, fictional observations for an offline ML workflow rehearsal."""
import datetime as dt
import random

from analytics import market_calendar as mc
from analytics import ml_readiness as MR
from analytics import ml_v3_audit as A
from analytics import ml_v3_pipeline as V3
from analytics import research_dataset as rd


def dataset(days=45, per_day=25):
    rng = random.Random(81273)
    dates, day = [], dt.date(2026, 2, 2)
    while len(dates) < days:
        if mc.is_trading_day(day):
            dates.append(day)
        day += dt.timedelta(days=1)
    frozen = dt.datetime.combine(dates[-1], dt.time(21), tzinfo=dt.timezone.utc)
    raw, scans = [], []
    for di, day in enumerate(dates):
        fired = dt.datetime.combine(day, dt.time(15), tzinfo=dt.timezone.utc)
        for k in range(per_day):
            oid = di * per_day + k + 1
            ticker = f"SYN{(di * per_day + k) % 160:03d}"
            score = rng.uniform(20, 95)
            positive = rng.random() < .45
            ret = rng.uniform(.01, .08) if positive else -rng.uniform(.01, .05)
            sid = fired - dt.timedelta(minutes=5)
            computed = MR.maturation_timing(fired)["due_at"] + dt.timedelta(hours=1)
            raw.append({"id": oid, "ticker": ticker, "fired_at": fired, "created_at": fired,
                        "signal_type": "hsf_scanner", "setup_score": score / 2,
                        "ai_confidence": rng.uniform(20, 80), "prebreakout_prob": 30,
                        "raw_signal": {"hsf_score": score, "score_version": "1.0", "status": "WATCH",
                                       "primary_setup": "breakout", "source_scan_id": sid.isoformat(),
                                       "prebreakout_model_version": "synthetic-v1"},
                        "indicators": {"signals": ["breakout"], "n_signals": 1, "chg_pct": 2, "gap_pct": 1},
                        "return_1d": ret / 3, "return_3d": ret / 2, "return_5d": ret,
                        "mfe_5d": .06 if positive else .025, "mae_5d": -.01 if positive else -.03,
                        "benchmark_return_1d": .001, "benchmark_return_3d": .002, "benchmark_return_5d": .003,
                        "benchmark_computed_at": computed, "outcome_computed_at": computed})
            scans.append({"observation_id": f"synthetic-{oid}", "symbol": ticker,
                          "context": "scheduled:us_market", "timestamp": sid, "created_at": sid,
                          "record": {"scan_timestamp": sid.isoformat(), "market": {"price": 20, "volume": 1000000},
                                     "indicators": {"rvol": rng.uniform(1, 4), "gap_pct": 1, "chg_pct": 2},
                                     "scanners": [{"name": "breakout", "score": score / 2}],
                                     "research_metadata": {"row_features": {"trend_10d_pct": rng.uniform(-5, 5),
                                                                                "trend_20d_pct": rng.uniform(-5, 5)},
                                                           "rank_at_observation": k + 1}}})
    raw = V3.as_of(raw, frozen)
    built = rd.build_dataset(raw, scans)
    return {"entry": {"dataset_version": "synthetic-rehearsal-v1", "created_at": frozen,
                       "fingerprint": built["metadata"]["fingerprint"], "observation_ids": built["observation_ids"]},
            "members": built["observation_ids"], "frozen_at": frozen, "built": built,
            "audit_rows": A.build_audit_rows(built["records"], raw, unit="signal_day"),
            "raw_rows": raw, "scan_rows": scans}
