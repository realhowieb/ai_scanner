"""Calibration analysis of the 15-min BTC outcome log — why is paper P&L negative?

Read-only against Neon (btc_outcomes). Answers three questions:
  A. Is Kalshi's implied price well-calibrated? (reliability curve + Brier)
  B. Is the engine's win-probability calibrated, or overconfident?
  C. Do any engine features predict the outcome beyond Kalshi's price?
Plus the paper-trade P&L breakdown. Run via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLS = ("pred_direction", "pred_confidence", "pred_win_prob", "kalshi_yes_pct",
        "result_up", "bet_side", "bet_price", "bet_pnl", "features")


def _rows():
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        print("no Neon connection")
        return []
    cur = conn.cursor()
    cur.execute(
        f"SELECT {', '.join(COLS)} FROM btc_outcomes "
        "WHERE settled = TRUE AND result_up IS NOT NULL"
    )
    raw = cur.fetchall() or []
    cur.close()
    out = []
    for r in raw:
        out.append(dict(r) if isinstance(r, dict) else dict(zip(COLS, r)))
    return out


def _brier(pairs):
    """pairs of (prob 0-1, outcome 0/1) → mean squared error."""
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _reliability(pairs, bins=5):
    """Bucket (prob, outcome) into `bins` and print predicted vs actual."""
    if not pairs:
        print("  (no data)")
        return
    width = 1.0 / bins
    print(f"  {'bucket':<12}{'n':>5}{'predicted':>12}{'actual':>10}")
    for i in range(bins):
        lo, hi = i * width, (i + 1) * width
        grp = [(p, y) for p, y in pairs if (lo <= p < hi or (i == bins - 1 and p == hi))]
        if not grp:
            continue
        pred = sum(p for p, _ in grp) / len(grp)
        act = sum(y for _, y in grp) / len(grp)
        print(f"  {lo:.0%}-{hi:.0%}{'':<4}{len(grp):>5}{pred:>11.0%}{act:>10.0%}")


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def main() -> None:
    rows = _rows()
    print("=" * 72)
    print(f"BTC 15-min calibration — {len(rows)} settled windows")
    print("=" * 72)
    if not rows:
        return

    up_rate = sum(1 for r in rows if r["result_up"]) / len(rows)
    print(f"base rate: BTC finished up {up_rate:.0%} of windows\n")

    # --- A. Kalshi calibration ---
    kal = [(float(r["kalshi_yes_pct"]) / 100.0, 1 if r["result_up"] else 0)
           for r in rows if r["kalshi_yes_pct"] is not None]
    print("A. KALSHI implied price — reliability (predicted should ≈ actual):")
    _reliability(kal)
    kb = _brier(kal)
    print(f"  Brier: {kb:.3f}  (0.25 = coin-flip; lower = sharper)\n")

    # --- B. Engine win-probability calibration (predicted-direction wins?) ---
    eng = []
    for r in rows:
        wp, d = r["pred_win_prob"], r["pred_direction"]
        if wp is None or d not in ("up", "down"):
            continue
        won = (d == "up") == bool(r["result_up"])
        eng.append((float(wp) / 100.0, 1 if won else 0))
    print(f"B. ENGINE win-probability — reliability ({len(eng)} directional calls):")
    _reliability(eng)
    eb = _brier(eng)
    if eb is not None:
        avg_pred = sum(p for p, _ in eng) / len(eng)
        avg_act = sum(y for _, y in eng) / len(eng)
        print(f"  Brier: {eb:.3f} · avg predicted {avg_pred:.0%} vs actual {avg_act:.0%} "
              f"→ {'OVERCONFIDENT' if avg_pred > avg_act + 0.05 else 'roughly calibrated'}\n")

    # --- C. Feature correlations with the outcome (result_up) ---
    print("C. FEATURE correlation with outcome (BTC up), |r| desc:")
    ys = [1.0 if r["result_up"] else 0.0 for r in rows]
    feat_keys = set()
    for r in rows:
        if isinstance(r.get("features"), dict):
            feat_keys.update(k for k, v in r["features"].items() if isinstance(v, (int, float)))
    corrs = []
    for k in feat_keys:
        xs, yy = [], []
        for r, y in zip(rows, ys):
            f = r.get("features") or {}
            v = f.get(k)
            if isinstance(v, (int, float)):
                xs.append(float(v))
                yy.append(y)
        c = _pearson(xs, yy)
        if c is not None:
            corrs.append((k, c, len(xs)))
    for k, c, n in sorted(corrs, key=lambda t: abs(t[1]), reverse=True):
        print(f"  {k:<20}{c:+.3f}   (n={n})")

    # --- D. Paper-trade P&L ---
    bets = [r for r in rows if r["bet_side"] and r["bet_price"] is not None]
    print(f"\nD. PAPER TRADES — {len(bets)} bets:")
    if bets:
        wins = sum(1 for r in bets if (r["bet_pnl"] or 0) > 0)
        pnl = sum(r["bet_pnl"] or 0 for r in bets)
        staked = sum(r["bet_price"] for r in bets)
        for side in ("YES", "NO"):
            sb = [r for r in bets if r["bet_side"] == side]
            if sb:
                spnl = sum(r["bet_pnl"] or 0 for r in sb)
                print(f"  {side}: {len(sb)} bets, P&L {spnl:+.2f}u")
        print(f"  total: {wins}/{len(bets)} won · P&L {pnl:+.2f}u · "
              f"ROI {(pnl / staked * 100 if staked else 0):+.1f}%")
        avg_entry = staked / len(bets)
        print(f"  avg entry price paid: {avg_entry:.0%}")


if __name__ == "__main__":
    main()
