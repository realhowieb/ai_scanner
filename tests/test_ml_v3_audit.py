"""Deterministic tests for the ML v3 leakage-safe walk-forward audit.

No live market API, no database: rows are built with tests/research_fixtures.
"""
from __future__ import annotations

import datetime as dt
import random

import numpy as np
import pytest
from research_fixtures import opp, scan

from analytics import market_calendar as mc
from analytics import ml_v3_audit as A
from analytics import ml_v3_pipeline as P
from analytics import research_dataset as rd
from analytics import research_schema as rs

pytest.importorskip("sklearn")


def _trading_days(start: dt.date, n: int):
    out, d = [], start
    while len(out) < n:
        if mc.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _dataset(n_days=30, per_day=8, seed=7, bench=True, start=dt.date(2026, 7, 1)):
    rng = random.Random(seed)
    rows, scans, oid = [], [], 1
    for d in _trading_days(start, n_days):
        for k in range(per_day):
            t = f"T{rng.randrange(60):02d}"
            score = float(round(rng.uniform(20, 95)))  # production scores are integers
            r5 = rng.gauss((score - 55) / 1000, 0.03)
            fired = f"{d.isoformat()}T{14 + k % 5:02d}:{(k * 7) % 60:02d}:00"
            rows.append(opp(oid, t, fired, score=score, scored=True, r1=r5 / 3, r3=r5 / 2, r5=r5,
                            mfe=abs(r5) + 0.01, mae=-0.015, prob=rng.uniform(5, 40),
                            b1=0.001 if bench else None, b3=0.002 if bench else None, b5=0.003 if bench else None))
            sts = (dt.datetime.fromisoformat(fired) - dt.timedelta(minutes=20)).isoformat()
            scans.append(scan(t, sts, trend20=rng.gauss(0, 5)))
            oid += 1
    return rows, scans


def _rows(unit="signal_day", **kw):
    raw, scans = _dataset(**kw)
    recs = rd.build_records(raw, scans)
    return A.build_audit_rows(recs, raw, unit=unit), raw, scans


# ----------------------------------------------------------------------------- chronology / purge / embargo
def test_folds_are_chronological_and_expanding():
    rows, _, _ = _rows()
    data = A.eligible(rows, 5)
    folds = A.walk_forward_folds(data, 5, min_train=40, min_val=20)
    assert len(folds) >= 2
    by_id = {r.observation_id: r for r, _ in data}
    prev_train = 0
    for f in folds:
        vstart = dt.date.fromisoformat(f.validation_start)
        assert all(by_id[i].entry_day < vstart for i in f.train_ids)
        assert all(by_id[i].entry_day >= vstart for i in f.val_ids)
        assert len(f.train_ids) >= prev_train
        prev_train = len(f.train_ids)
    # validation blocks never overlap
    seen = set()
    for f in folds:
        assert not (seen & set(f.val_ids))
        seen |= set(f.val_ids)


@pytest.mark.parametrize("h", [1, 3, 5])
def test_no_training_label_reads_a_validation_period_bar(h):
    rows, _, _ = _rows()
    data = A.eligible(rows, h)
    folds = A.walk_forward_folds(data, h, min_train=40, min_val=20, embargo=0)
    assert folds
    by_id = {r.observation_id: r for r, _ in data}
    for f in folds:
        vstart = dt.date.fromisoformat(f.validation_start)
        for i in f.train_ids:
            assert by_id[i].window_end[h] < vstart  # label's last bar is before validation starts


def test_purge_drops_overlapping_outcome_windows():
    rows, _, _ = _rows()
    vstart = sorted({r.entry_day for r in rows})[15]
    kept, purged, emb = A.purge_train(rows, vstart, 5, embargo=0)
    overlapping = [r for r in rows if r.entry_day < vstart and r.window_end[5] >= vstart]
    assert purged == len(overlapping) > 0
    assert emb == 0
    assert not ({r.observation_id for r in kept} & {r.observation_id for r in overlapping})


def test_embargo_adds_an_extra_gap():
    rows, _, _ = _rows()
    vstart = sorted({r.entry_day for r in rows})[15]
    k0, p0, e0 = A.purge_train(rows, vstart, 5, embargo=0)
    k2, p2, e2 = A.purge_train(rows, vstart, 5, embargo=2)
    assert p0 == p2 and e2 > 0 and len(k2) == len(k0) - e2
    cutoff = A._trading_days_before(vstart, 2)
    assert all(r.window_end[5] < cutoff for r in k2)


def test_longer_horizon_purges_more():
    rows, _, _ = _rows()
    vstart = sorted({r.entry_day for r in rows})[15]
    assert A.purge_train(rows, vstart, 5, 0)[1] > A.purge_train(rows, vstart, 1, 0)[1]


def test_fold_construction_is_reproducible_and_order_independent():
    rows, _, _ = _rows()
    data = A.eligible(rows, 5)
    a = A.walk_forward_folds(data, 5, min_train=40, min_val=20)
    shuffled = list(data)
    random.Random(3).shuffle(shuffled)
    b = A.walk_forward_folds(shuffled, 5, min_train=40, min_val=20)
    assert [(f.validation_start, sorted(f.train_ids), sorted(f.val_ids)) for f in a] == \
           [(f.validation_start, sorted(f.train_ids), sorted(f.val_ids)) for f in b]


def test_holdout_is_excluded_from_every_fold_and_refused_when_small():
    rows, _, _ = _rows(n_days=40)
    data = A.eligible(rows, 5)
    hold = A.final_holdout(data, 5, min_rows=40, min_days=5, min_remaining_folds=2, min_train=40, min_val=20)
    assert hold["valid"]
    folds = A.walk_forward_folds(data, 5, exclude_ids=hold["ids"], min_train=40, min_val=20)
    for f in folds:
        assert not (set(hold["ids"]) & (set(f.train_ids) | set(f.val_ids)))
    tiny = A.final_holdout(data[:50], 5, min_rows=100, min_days=5)
    assert tiny["valid"] is False and "need" in tiny["reason"]


def test_walk_forward_has_no_future_records_in_training():
    rows, _, _ = _rows()
    data = A.eligible(rows, 5)
    folds = A.walk_forward_folds(data, 5, min_train=40, min_val=20)
    res = A.run_walk_forward(data, folds, A.Majority, 5)
    by_id = {r.observation_id: r for r, _ in data}
    for f in folds:
        latest_train_obs = max(by_id[i].observed_at for i in f.train_ids)
        earliest_val_obs = min(by_id[i].observed_at for i in f.val_ids)
        assert latest_train_obs < earliest_val_obs
    assert res["summary"]["roc_auc"]["mean"] == 0.5  # a constant can't rank


# ----------------------------------------------------------------------------- unit / duplicates
def test_signal_day_unit_keeps_first_observation_per_ticker_day():
    raw = [opp(1, "AAA", "2026-07-06T14:00:00", scored=True, r1=.01, r3=.01, r5=.01),
           opp(2, "AAA", "2026-07-06T16:00:00", scored=True, r1=.01, r3=.01, r5=.01),
           opp(3, "AAA", "2026-07-07T14:00:00", scored=True, r1=.01, r3=.01, r5=.01)]
    recs = rd.build_records(raw, [])
    day = A.build_audit_rows(recs, raw, unit="signal_day")
    obs = A.build_audit_rows(recs, raw, unit="observation")
    assert [r.observation_id for r in day] == [1, 3]
    assert len(obs) == 3


# ----------------------------------------------------------------------------- features vs labels
def test_feature_snapshot_never_carries_labels():
    rows, _, _ = _rows()
    for r in rows:
        assert not any(rs.looks_like_outcome(k) for k in r.features)
        assert set(r.labels) <= set(rs.label_names())
    enc = A.Encoder(*A.feature_set()).fit(rows)
    assert not any(rs.looks_like_outcome(c.split("=")[0]) for c in enc.columns)


def test_scan_features_only_from_records_written_before_observation():
    raw = [opp(1, "AAA", "2026-07-06T15:00:00", scored=True, r5=.01, r1=.01, r3=.01)]
    late = scan("AAA", "2026-07-06T14:50:00", written="2026-07-06T15:05:00", price=99.0)  # written after T
    early = scan("AAA", "2026-07-06T14:30:00", price=11.0)
    recs = rd.build_records(raw, [late, early])
    row = A.build_audit_rows(recs, raw)[0]
    assert row.features["price"] == 11.0
    assert row.join_lag_s is not None and row.join_lag_s >= 0
    assert A.join_integrity([row])["negative_lag_violations"] == 0


def test_encoder_is_fitted_on_training_rows_only():
    rows, _, _ = _rows()
    tr, va = rows[:50], rows[50:]
    for r in va:
        r.features["ema_cross"] = "only_in_validation"
    enc = A.Encoder(*A.feature_set()).fit(tr)
    assert not any("only_in_validation" in c for c in enc.columns)
    X = enc.transform(va, impute=True)
    assert X.shape == (len(va), len(enc.columns)) and not np.isnan(X).any()


# ----------------------------------------------------------------------------- labels / benchmark / MFE
def test_label_construction_and_benchmark_alignment():
    lab = {"return_5d": 0.03, "benchmark_return_5d": 0.01, "excess_return_5d": 0.02, "mfe_5d": 0.05, "mae_5d": -0.01}
    assert A.LABELS["A_return_gt_0"].fn(lab, 5) == 1
    assert A.LABELS["B_return_ge_4pct"].fn(lab, 5) == 0
    assert A.LABELS["C_excess_gt_0"].fn(lab, 5) == 1
    assert A.LABELS["D_excess_ge_2pct"].fn(lab, 5) == 1
    assert A.LABELS["E_abs_and_rel_win"].fn(lab, 5) == 1
    assert A.LABELS["F_clean_path_5d"].fn(lab, 5) == 1
    assert A.LABELS["F_clean_path_5d"].fn({**lab, "mae_5d": -0.03}, 5) == 0
    # benchmark-relative labels are undefined (None), never 0, without SPY
    assert A.LABELS["C_excess_gt_0"].fn({"return_5d": 0.03, "excess_return_5d": None}, 5) is None
    # MFE/MAE exist only at 5 days
    assert A.LABELS["F_clean_path_5d"].fn(lab, 3) is None


def test_excess_return_comes_from_canonical_outcome_record():
    raw = [opp(1, "AAA", "2026-07-06T14:00:00", scored=True, r1=.01, r3=.02, r5=.03, b1=.0, b3=.01, b5=.02)]
    row = A.build_audit_rows(rd.build_records(raw, []), raw)[0]
    assert row.labels["excess_return_5d"] == pytest.approx(0.01)
    assert row.labels["excess_return_3d"] == pytest.approx(0.01)


def test_missing_mfe_mae_is_reported_not_filled():
    raw = [opp(1, "AAA", "2026-07-06T14:00:00", scored=True, r1=.01, r3=.01, r5=.01, mfe=None, mae=None)]
    row = A.build_audit_rows(rd.build_records(raw, []), raw)[0]
    assert row.labels["mfe_5d"] is None and row.labels["mae_5d"] is None
    fin = A.financial_metrics([row], 5)
    assert fin["mfe_count"] == 0 and fin["median_mfe"] is None


def test_eligible_requires_matured_and_certified():
    raw = [opp(1, "AAA", "2026-07-06T14:00:00", scored=False),
           opp(2, "BBB", "2026-07-06T14:00:00", scored=True, r1=None, r3=None, r5=None),
           opp(3, "CCC", "2026-07-06T14:00:00", scored=True, r1=.01, r3=.01, r5=-.01)]
    rows = A.build_audit_rows(rd.build_records(raw, []), raw)
    assert [(r.observation_id, y) for r, y in A.eligible(rows, 5)] == [(3, 0)]


# ----------------------------------------------------------------------------- score buckets
def test_score_buckets_use_canonical_definitions_and_flag_inversions():
    rows, _, _ = _rows()
    cal = A.score_calibration(rows)
    buckets = [b["bucket"] for b in cal if b["horizon"] == 5]
    assert buckets == [f"{lo}-{hi}" for lo, hi in A.oi.SCORE_BUCKETS]
    assert sum(b["sample_size"] for b in cal if b["horizon"] == 5) == sum(1 for r in rows if r.matured(5))
    assert all("monotonicity" in b for b in cal)


# ----------------------------------------------------------------------------- fingerprint / immutability
def test_slice_fingerprint_is_deterministic_and_sensitive():
    rows, raw, scans = _rows()
    again = A.build_audit_rows(rd.build_records(raw, scans), raw)
    assert A.slice_fingerprint(rows) == A.slice_fingerprint(again)
    raw2 = [dict(r) for r in raw]
    raw2[0]["return_5d"] = 0.5
    changed = A.build_audit_rows(rd.build_records(raw2, scans), raw2)
    assert A.slice_fingerprint(changed) != A.slice_fingerprint(rows)


def test_as_of_freeze_view_is_immutable_against_later_maturation_and_backfill():
    raw, _ = _dataset(n_days=10, per_day=3, bench=False)
    raw.append(opp(999, "ZZZ", "2026-07-14T14:00:00", scored=False))  # pending at freeze time
    frozen_at = max(r["outcome_computed_at"] for r in raw if r["outcome_computed_at"]) + dt.timedelta(seconds=1)
    members = [r["id"] for r in raw]
    fp_frozen = rd.build_dataset(P.as_of(raw, frozen_at), [], members=members)["metadata"]["fingerprint"]
    later = [dict(r) for r in raw]
    # after the freeze: SPY backfill fills one row, and the pending row matures
    later[0].update(benchmark_return_1d=0.1, benchmark_return_3d=0.2, benchmark_return_5d=0.3,
                    benchmark_computed_at=frozen_at + dt.timedelta(days=1))
    later[-1].update(outcome_computed_at=frozen_at + dt.timedelta(days=2), return_1d=0.1, return_3d=0.1,
                     return_5d=0.2)
    assert rd.build_dataset(later, [], members=members)["metadata"]["fingerprint"] != fp_frozen
    view = P.as_of(later, frozen_at)
    assert view[0]["benchmark_return_5d"] is None
    assert view[-1]["outcome_computed_at"] is None and view[-1]["return_5d"] is None
    assert rd.build_dataset(view, [], members=members)["metadata"]["fingerprint"] == fp_frozen


# ----------------------------------------------------------------------------- metrics / models
def test_score_model_probability_map_is_fitted_on_training_fold():
    rows, _, _ = _rows()
    data = A.eligible(rows, 5)
    folds = A.walk_forward_folds(data, 5, min_train=40, min_val=20)
    fac = lambda: A.ScoreModel("hsf", "hsf_score")  # noqa: E731
    r1 = A.run_walk_forward(data, folds, fac, 5)
    r2 = A.run_walk_forward(data, folds, fac, 5)
    assert r1["summary"] == r2["summary"]
    assert all(0 <= p <= 1 for p in r1["predictions"]["p"])


def test_unscorable_rows_are_counted_not_silently_dropped():
    rows, _, _ = _rows()
    for r in rows[::2]:
        r.features["prebreakout_prob"] = None
    data = A.eligible(rows, 5)
    folds = A.walk_forward_folds(data, 5, min_train=40, min_val=20)
    res = A.run_walk_forward(data, folds, lambda: A.ScoreModel("pb", "prebreakout_prob"), 5)
    assert res["excluded_unscorable_rows"] > 0


def test_random_baseline_is_deterministic():
    rows, _, _ = _rows()
    assert list(A.RandomBaseline().predict(rows)) == list(A.RandomBaseline().predict(rows))


def test_regime_uses_only_prior_closes():
    closes = {d: 100.0 + i for i, d in enumerate(_trading_days(dt.date(2026, 5, 1), 60))}
    day = sorted(closes)[40]
    base = A.spy_regime_by_day(closes, [day])[day]
    spiked = dict(closes)
    spiked[day] = 1.0  # the entry day's own close must not matter
    for d in [x for x in closes if x > day]:
        spiked[d] = 1.0
    assert A.spy_regime_by_day(spiked, [day])[day] == base


def test_legacy_future_breakout_label_matches_reference_definition():
    pd = pytest.importorskip("pandas")
    from ml_prebreakout import add_future_breakout_label

    ts = pd.Timestamp("2026-07-01", tz="UTC")
    df = pd.DataFrame({"Symbol": ["A"] * 5 + ["B"] * 3,
                       "Timestamp": [ts + pd.Timedelta(minutes=10 * i) for i in range(8)],
                       "IsBreakout": [0, 0, 0, 1, 0, 1, 0, 0]})
    ours = A.future_breakout_label(df).sort_values(["Symbol", "Timestamp"]).reset_index(drop=True)
    ref = add_future_breakout_label(df.copy())
    assert list(ours["FutureBreakout"]) == list(ref["FutureBreakout"])
    # the reference reads the next 3 rows of the whole frame: A's last-but-one row is
    # labelled 1 from B's breakout (cross-symbol bleed), and the reproduction keeps it
    assert ours.loc[3, "FutureBreakout"] == 1 and ours.loc[3, "Symbol"] == "A"
