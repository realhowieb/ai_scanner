#!/usr/bin/env python3
"""Run 55 — research evidence & signal effectiveness runner (READ-ONLY).

Loads production research observations and their outcome records, runs the pure
`analytics.signal_evidence.analyze`, and writes:

    artifacts/research/run55_signal_effectiveness.json
    artifacts/research/run55_signal_effectiveness.md

It never writes observations, outcomes, scans, or scanners, and changes no
scoring, ranking, tier, cohort, or outcome logic. Verdict rules are pre-declared
in `analytics.signal_evidence.CRITERIA`.

    python -m scripts.analyze_signal_effectiveness [--out DIR] [--input FILE.json]

`--input` replays a saved {"observations": [...], "outcomes_by_id": {...}}
snapshot instead of reading the database (reproducibility).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from analytics.signal_evidence import (
    COHORTS,
    HORIZONS,
    PRIMARY_HORIZON,
    TOD_BUCKETS,
    analyze,
)

ROOT = Path(__file__).resolve().parents[1]


def _pct(x: Any, nd: int = 2) -> str:
    return "—" if x is None else f"{x * 100:+.{nd}f}%"


def _rate(x: Any) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _ci(ci, fmt=_pct) -> str:
    lo, hi = (list(ci or []) + [None, None])[:2]
    return "—" if lo is None or hi is None else f"[{fmt(lo)}, {fmt(hi)}]"


def _stat_row(label: str, s: Dict[str, Any], excursions: bool = False) -> str:
    base = (f"| {label} | {s['n']} | {_rate(s['win_rate'])} {_ci(s['win_rate_ci'], _rate)} | "
            f"{_pct(s['mean'], 3)} {_ci(s['mean_ci'], lambda v: _pct(v, 3))} | {_pct(s['median'], 3)} | "
            f"{_pct(s['sd'], 3)} | {_pct(s['p25'], 3)} / {_pct(s['p75'], 3)} | "
            f"{'—' if s['payoff_ratio'] is None else s['payoff_ratio']} |")
    if excursions:
        base += (f" {_pct(s.get('mfe_mean'), 3)} / {_pct(s.get('mfe_median'), 3)} | "
                 f"{_pct(s.get('mae_mean'), 3)} / {_pct(s.get('mae_median'), 3)} |")
    return base


_HDR = ("| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |")
_HDR_X = _HDR + " MFE mean / median | MAE mean / median |"


def _hdr(excursions: bool) -> List[str]:
    h = _HDR_X if excursions else _HDR
    return [h, "|" + "---|" * (h.count("|") - 1)]


def _cmp_row(label: str, c: Dict[str, Any]) -> str:
    return (f"| {label} | {c['a_n']} vs {c['b_n']} | {c['clusters']} | {_pct(c['diff_mean'], 3)} "
            f"{_ci(c['diff_mean_ci'], lambda v: _pct(v, 3))} | {_pct(c['diff_win_rate'], 1)} "
            f"{_ci(c['diff_win_ci'], lambda v: _pct(v, 1))} | {_pct(c.get('mde'), 3)} | "
            f"{'yes' if c.get('powered') else 'no'} | {c.get('additional_needed_per_arm', '—')} | "
            f"{c['evidence']} |")


_CMP_HDR = ["| Comparison | N (A vs B) | Scan runs | Δ mean [95% cluster CI] | Δ win rate [CI] | "
            "MDE (80%) | Powered | + N/arm needed | Evidence |",
            "|---|---|---|---|---|---|---|---|---|"]


def render_markdown(r: Dict[str, Any]) -> str:
    v, rd, coh, mono = r["verdict"], r["readiness"], r["cohorts"], r["score_monotonicity"]
    L: List[str] = ["# Run 55 — Research Evidence & Signal Effectiveness", "",
                    f"Generated {r['generated_at']} · schema `{r['schema']}` · READ-ONLY", ""]

    L += ["## 1. Executive verdict", "",
          f"**Overall verdict: {v['verdict']}** · evidence quality **{v['evidence_quality']}**", "",
          f"- Reason: {v['reason']}",
          f"- Readiness: **{rd['verdict']}**",
          f"- Higher scores predict better outcomes: **{mono['verdict']}** ({mono['reason']})",
          f"- Tiers separate outcome quality: **{r['tiers']['verdict']}**",
          f"- Run 56 recommendation: **{r['run56']['code']}. {r['run56']['action']}** — {r['run56']['reason']}",
          "", "Verdict rules were fixed in `analytics/signal_evidence.py` (`CRITERIA`, "
          "`primary_verdict`) before the production data was analyzed.", ""]

    L += ["## 2. Dataset / readiness", "",
          "| Metric | Value |", "|---|---|"]
    for k in ("total_observations", "explicitly_tagged", "legacy_inferred", "unique_symbols",
              "unique_scan_runs", "matured_observations", "unmatured_observations",
              "retirement_eligible_unmatured", "matured_outcome_records",
              "directional_return_stored_coverage", "mfe_coverage", "mae_coverage",
              "duplicate_outcomes", "conflicting_outcomes", "orphan_outcomes",
              "point_in_time_violations", "invalid_observations", "invalid_outcomes",
              "extreme_returns_abs_gt_50pct", "cohort_overlap_dropped"):
        L.append(f"| {k} | {rd[k]} |")
    L += ["", f"Explicit observations by cohort: {rd['observations_by_cohort_explicit']} · "
          f"all (incl. legacy): {rd['observations_by_cohort_all']}",
          f"Primary-population N at {PRIMARY_HORIZON}: {rd['primary_population_n_by_cohort']} · "
          f"direction N: {rd['direction_n_primary']}", "",
          "Matured observations by cohort × horizon (explicit):", "",
          "| Cohort | " + " | ".join(HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
    for c in COHORTS:
        L.append(f"| {c} | " + " | ".join(str(rd['coverage_by_cohort_horizon'][c][h]) for h in HORIZONS) + " |")
    L += ["", "Readiness gates:", "", "| Gate | Passed | If failed | Detail |", "|---|---|---|---|"]
    for g in rd["gates"]:
        L.append(f"| {g['gate']} | {'✅' if g['passed'] else '❌'} | {g['severity_if_failed']} | {g['detail']} |")
    L.append("")

    L += ["## 3. Cohort comparison", "",
          "Directional returns, winsorized per horizon at pooled "
          f"{r['criteria']['winsor_pct']}/{100 - r['criteria']['winsor_pct']} percentiles "
          f"(limits: {r['winsor_limits']}). Per-group CIs are i.i.d. approximations; "
          "differences use a cluster bootstrap over scan runs.", ""]
    for h in HORIZONS:
        b = coh["by_horizon"][h]
        ex = h == PRIMARY_HORIZON
        L += [f"### {h}", ""] + _hdr(ex)
        for c in COHORTS:
            L.append(_stat_row(c, b["cohorts"][c], ex))
        L += [""] + _CMP_HDR
        L.append(_cmp_row("CANDIDATE − CONTROL", b["candidate_minus_control"]))
        L.append(_cmp_row("CANDIDATE − NEAR_MISS", b["candidate_minus_near_miss"]))
        L.append(_cmp_row("NEAR_MISS − CONTROL", b["near_miss_minus_control"]))
        L.append("")

    L += ["## 4. Horizon comparison", "",
          "| Horizon | CAND−CTRL Δ mean | CI sign | Evidence | CAND−NM Δ mean | CI sign | NM−CTRL Δ mean | CI sign |",
          "|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        b = coh["by_horizon"][h]
        cc, cn, nc = b["candidate_minus_control"], b["candidate_minus_near_miss"], b["near_miss_minus_control"]
        L.append(f"| {h} | {_pct(cc['diff_mean'], 3)} | {cc['sign']} | {cc['evidence']} | "
                 f"{_pct(cn['diff_mean'], 3)} | {cn['sign']} | {_pct(nc['diff_mean'], 3)} | {nc['sign']} |")
    L.append("")

    spec = mono["bucket_spec"]
    L += ["## 5. Score monotonicity", "",
          f"**DO HIGHER SCORES PREDICT BETTER OUTCOMES? {mono['verdict']}** — {mono['reason']}", "",
          f"Population: {mono['population']}. Bucket mode: **{spec['mode']}** "
          f"({', '.join(spec['buckets'])}); scored N = {spec['scored_n']}; "
          f"score quantiles [min, p10, p25, p50, p75, p90, max] = {spec['score_quantiles']}.", ""]
    for h in HORIZONS:
        m = mono["by_horizon"][h]
        ex = h == PRIMARY_HORIZON
        L += [f"### {h} — Spearman {m['spearman']} {_ci(m['spearman_ci'], lambda x: f'{x:+.3f}')} "
              f"({m['sign']}), Pearson {m['pearson']}, adjacent violations {m['adjacent_violations']} "
              f"of {max(0, m['comparable_buckets'] - 1)}", ""] + _hdr(ex)
        for s in m["buckets"]:
            L.append(_stat_row(s["bucket"], s, ex))
        L.append("")

    t = r["tiers"]
    L += ["## 6. Tier effectiveness", "",
          f"**DO CURRENT TIERS SEPARATE OUTCOME QUALITY? {t['verdict']}**", ""]
    if not t.get("available"):
        L += [f"Not measurable: {t['reason']}.", ""]
    else:
        L += [f"Tier share at {PRIMARY_HORIZON}: {t['share']}", ""]
        for h in HORIZONS:
            L += [f"### {h}", ""] + _hdr(h == PRIMARY_HORIZON)
            for tier_name in ("WEAK", "DEVELOPING", "STRONG"):
                L.append(_stat_row(tier_name, t["by_horizon"][h][tier_name], h == PRIMARY_HORIZON))
            L += [""] + _CMP_HDR + [_cmp_row("STRONG − WEAK", t["by_horizon"][h]["strong_minus_weak"]), ""]

    d = r["direction"]
    tc = d["short_transform_check"]
    L += ["## 7. LONG vs SHORT", "", f"**{d['verdict']}** — {d['reason']}", "",
          f"SHORT transform check (stored directional_return == −raw for SHORT, == raw for LONG): "
          f"{tc['checked']} checked, {tc['mismatches']} mismatches; stored {tc['stored']}, "
          f"derived {tc['derived']}.", ""]
    for h in HORIZONS:
        L += [f"### {h}", ""] + _hdr(h == PRIMARY_HORIZON)
        for dn in ("LONG", "SHORT"):
            L.append(_stat_row(dn, d["by_horizon"][h][dn], h == PRIMARY_HORIZON))
        L.append("")

    tod = r["time_of_day"]
    L += ["## 8. Time-of-day", "", f"**{tod['verdict']}** — {tod['reason']}", "",
          f"Population: {tod['population']}. N per bucket at {PRIMARY_HORIZON}: {tod['n_by_bucket_primary']}", ""]
    prim = tod["by_horizon"][PRIMARY_HORIZON]
    L += [f"### {PRIMARY_HORIZON} — CANDIDATE by bucket", ""] + _hdr(True)
    for _lo, _hi, lab in TOD_BUCKETS:
        L.append(_stat_row(lab, prim[lab]["CANDIDATE"], True))
    L += ["", f"### {PRIMARY_HORIZON} — CONTROL by bucket", ""] + _hdr(True)
    for _lo, _hi, lab in TOD_BUCKETS:
        L.append(_stat_row(lab, prim[lab]["CONTROL"], True))
    L += ["", _CMP_HDR[0], _CMP_HDR[1]]
    for _lo, _hi, lab in TOD_BUCKETS:
        L.append(_cmp_row(f"{lab}: CAND − CTRL", prim[lab]["candidate_minus_control"]))
    L.append("")

    reg = r["regime"]
    L += ["## 9. Market regime", "", f"**{reg['verdict']}** (point-in-time regime present on "
          f"{_rate(reg['present_share'])} of primary observations). {reg.get('reason', '')}", ""]

    f = r["features"]
    L += ["## 10. Feature / conflict diagnostics", "",
          f"Population: {f['population']}. {f['multiple_comparisons_note']}", "",
          "| Feature | Class | " + " | ".join(f"Spearman {h} [CI]" for h in HORIZONS) +
          f" | {PRIMARY_HORIZON} winners mean / losers mean |",
          "|---|---|" + "---|" * (len(HORIZONS) + 1)]
    for name, fv in f["numeric"].items():
        cells = []
        for h in HORIZONS:
            x = fv["by_horizon"][h]
            cells.append("—" if x["spearman"] is None else
                         f"{x['spearman']:+.3f} {_ci(x['spearman_ci'], lambda y: f'{y:+.3f}')} (n={x['n']})")
        p = fv["by_horizon"][PRIMARY_HORIZON]
        L.append(f"| {name} | {fv['classification']} | " + " | ".join(cells) +
                 f" | {p['winners_mean']} / {p['losers_mean']} |")
    L += ["", "| Flag (with − without) | Class | " + " | ".join(HORIZONS) + " |",
          "|---|---|" + "---|" * len(HORIZONS)]
    for name, fv in f["flags"].items():
        cells = [f"{_pct(fv['by_horizon'][h]['diff_mean'], 3)} {_ci(fv['by_horizon'][h]['diff_mean_ci'], lambda y: _pct(y, 3))} "
                 f"({fv['by_horizon'][h]['a_n']}/{fv['by_horizon'][h]['b_n']})" for h in HORIZONS]
        L.append(f"| {name} | {fv['classification']} | " + " | ".join(cells) + " |")
    L += ["", "Not measurable (not persisted at observation time):", ""]
    L += [f"- **{k}**: {why}" for k, why in f["unavailable"].items()]
    L.append("")

    e = r["excursions"]
    L += ["## 11. MFE / MAE quality", "", f"Horizon {e['horizon']}. {e['note']}.", "",
          "| Group | N | MFE mean / median | MAE mean / median | MFE ÷ abs(MAE) | Strong move → reversal | "
          "Early adverse → winner | Early favourable → loser |", "|---|---|---|---|---|---|---|---|"]
    for label, x in list(e["by_cohort"].items()) + [(f"score {k}", v) for k, v in e["by_score_bucket"].items()]:
        L.append(f"| {label} | {x['n']} | {_pct(x['mfe_mean'], 3)} / {_pct(x['mfe_median'], 3)} | "
                 f"{_pct(x['mae_mean'], 3)} / {_pct(x['mae_median'], 3)} | {x['mfe_mae_ratio']} | "
                 f"{_rate(x['strong_move_then_reversal_share'])} | {_rate(x['early_adverse_then_winner_share'])} | "
                 f"{_rate(x['early_favourable_then_loser_share'])} |")
    L.append("")

    L += ["## 12. Statistical power", "",
          f"Meaningful effect (pre-declared): {r['criteria']['meaningful_effect']:.2%}. MDE is the "
          "difference detectable with 80% power at α = 0.05, from the cluster-bootstrap SE. "
          "“+ N/arm needed” scales the smaller arm by (MDE / meaningful effect)².", "",
          "| Horizon | CAND−CTRL MDE | Powered | + N/arm | CAND−NM MDE | Powered | + N/arm |",
          "|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        cc, cn = coh["by_horizon"][h]["candidate_minus_control"], coh["by_horizon"][h]["candidate_minus_near_miss"]
        L.append(f"| {h} | {_pct(cc.get('mde'), 3)} | {'yes' if cc['powered'] else 'no'} | "
                 f"{cc.get('additional_needed_per_arm', '—')} | {_pct(cn.get('mde'), 3)} | "
                 f"{'yes' if cn['powered'] else 'no'} | {cn.get('additional_needed_per_arm', '—')} |")
    L.append("")

    L += ["## 13. Limitations", "",
          "- Horizons count minute bars, not wall-clock minutes; sparse symbols' “+60m” can span hours.",
          "- CONTROL rows have no direction and are measured long; every CANDIDATE/NEAR_MISS row's first "
          "scanner is `breakout` (long), so SHORT is effectively unobserved.",
          "- Outcomes matured before Run 53A lack stored directional_return/MFE/MAE; directional_return "
          "is derived from raw_return for those, MFE/MAE cannot be recovered.",
          "- Candidates and near-misses are adjacent in score rank (top-N vs just below), which restricts "
          "the score range within the scored population.",
          "- Observations of the same symbol in consecutive runs overlap in time; the cluster bootstrap "
          "treats scan runs, not symbols, as independent.",
          "- Tiers, conflicts, EMA/RSI and PreBreakout are not persisted on observations (Parts 4 and 8).",
          "- Point-in-time market regime is not captured (Part 9).", ""]

    rec = r["run56"]
    L += ["## 14. Run 56 recommendation", "",
          f"**{rec['code']}. {rec['action']}** — {rec['reason']}", ""]
    L += [f"- {n}" for n in rec["notes"]]
    L += ["", "Not implemented in Run 55.", ""]
    return "\n".join(L)


def _load(input_path: str | None):
    if input_path:
        data = json.loads(Path(input_path).read_text())
        return data.get("observations") or [], data.get("outcomes_by_id") or {}
    from scripts.audit_research_cohorts import _load_live
    return _load_live()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 55 signal effectiveness (read-only)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "research"))
    ap.add_argument("--input", default=None, help="replay a saved snapshot instead of the DB")
    ap.add_argument("--snapshot", action="store_true",
                    help="also write the loaded input snapshot (for exact replay)")
    args = ap.parse_args()

    observations, outcomes = _load(args.input)
    report = analyze(observations, outcomes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run55_signal_effectiveness.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "run55_signal_effectiveness.md").write_text(render_markdown(report))
    if args.snapshot:
        (out / "run55_input_snapshot.json").write_text(json.dumps(
            {"observations": observations, "outcomes_by_id": outcomes}, default=str))
    v = report["verdict"]
    rd = report["readiness"]
    print("RUN 55 — SIGNAL EFFECTIVENESS (read-only)")
    print(f"observations={rd['total_observations']} explicit={rd['explicitly_tagged']} "
          f"matured={rd['matured_observations']} readiness={rd['verdict']}")
    print(f"VERDICT: {v['verdict']} ({v['evidence_quality']}) — {v['reason']}")
    print(f"RUN 56: {report['run56']['code']}. {report['run56']['action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
