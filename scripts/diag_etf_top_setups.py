"""Read-only: how many ETFs land in Today's Top setups, and why.

For the latest scheduled full-market scan, prints the Top setups board (strong HSF 75+
then the HSF 40+ fill, 10 rows like /v1/today) with each row's sector, score parts,
change %, last price and signals. Then for every market scan of the last 15 sessions,
counts ETFs (sectors.csv sector "ETF") in that board, in the whole ranked list, and in
the raw scan universe, so the ETF share of Top setups can be compared with the ETF share
of what was scanned. Prints tickers and public market numbers only.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOP_N = 10


def board(df):
    from ui.market_scans import top_setups
    from ui.opportunities import HSF_STRONG_MIN
    from ui.recap import RECAP_MIN_SCORE

    strong = top_setups(df, n=TOP_N, minimum_score=HSF_STRONG_MIN)
    room = TOP_N - len(strong)
    if room > 0:
        have = {r["ticker"] for r in strong}
        pool = top_setups(df, n=TOP_N * 2, minimum_score=RECAP_MIN_SCORE)
        strong += [r for r in pool if r["ticker"] not in have][:room]
    return strong


def main() -> int:
    from api.sectors import sector_for
    from ui.market_scans import _list_runs_uncached, _run_df_uncached, top_setups

    def is_etf(t):
        return sector_for(t) == "ETF"

    runs = _list_runs_uncached()
    if not runs:
        print("no market runs")
        return 1
    latest = runs[0]
    df = _run_df_uncached(int(latest["id"]))
    print(f"latest scan run={latest['id']} at={latest.get('created_at')} rows={len(df)}")
    print("rank ticker  sector        score sig  mod  mom  fade  chg%     last     rvol  signals")
    for i, r in enumerate(board(df), 1):
        c = r.get("score_components") or {}
        print(f"{i:>4} {r['ticker']:<7} {str(sector_for(r['ticker']) or '-')[:13]:<13} {r['score']:>5} "
              f"{c.get('signals_component', 0):>4.0f} {c.get('model_component', 0):>4.0f} "
              f"{c.get('momentum_component', 0):>4.0f} {c.get('fading_penalty', 0):>4.0f} "
              f"{(r.get('chg_pct') or 0):>7.1f} {(r.get('last') or 0):>8.2f} {(r.get('rvol') or 0):>6.1f}  "
              f"{','.join(r.get('signals') or [])}")

    ranked = top_setups(df, n=10_000, minimum_score=40)
    print("\nlatest ranked (HSF 40+) top 30 with ETF flag:")
    print(" ".join(f"{r['ticker']}{'*' if is_etf(r['ticker']) else ''}:{r['score']}" for r in ranked[:30]))

    print("\nper scan: board ETFs/10 | ranked 40+ ETFs/total | 75+ ETFs/total | universe ETF share")
    totals = Counter()
    days = []
    for run in runs:
        day = str(run.get("created_at"))[:10]
        if day not in days:
            days.append(day)
        if len(days) > 15:
            break
        try:
            d = _run_df_uncached(int(run["id"]))
        except Exception as e:  # noqa: BLE001
            print(f"  run {run['id']} load failed: {type(e).__name__}")
            continue
        if d is None or d.empty:
            continue
        b = board(d)
        rk = top_setups(d, n=10_000, minimum_score=40)
        st = [r for r in rk if r["score"] >= 75]
        col = next((c for c in ("Ticker", "ticker", "Symbol", "symbol") if c in d.columns), None)
        uni = set(d[col].astype(str).str.upper()) if col else set()
        u_etf = sum(1 for t in uni if is_etf(t))
        b_etf = sum(1 for r in b if is_etf(r["ticker"]))
        r_etf = sum(1 for r in rk if is_etf(r["ticker"]))
        s_etf = sum(1 for r in st if is_etf(r["ticker"]))
        totals.update(board=len(b), board_etf=b_etf, ranked=len(rk), ranked_etf=r_etf,
                      strong=len(st), strong_etf=s_etf, uni=len(uni), uni_etf=u_etf,
                      lead_etf=int(bool(b) and is_etf(b[0]["ticker"])), scans=1)
        top_etfs = ",".join(r["ticker"] for r in b if is_etf(r["ticker"]))
        print(f"  {str(run.get('created_at'))[:16]} board {b_etf}/{len(b)} | ranked {r_etf}/{len(rk)} | "
              f"75+ {s_etf}/{len(st)} | universe {u_etf}/{len(uni)}  [{top_etfs}]")

    t = totals
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"  # noqa: E731
    print(f"\nTOTAL scans={t['scans']} board ETF share={pct(t['board_etf'], t['board'])} "
          f"ETF leads board in {t['lead_etf']}/{t['scans']} | ranked 40+ ETF share={pct(t['ranked_etf'], t['ranked'])} "
          f"| 75+ ETF share={pct(t['strong_etf'], t['strong'])} | universe ETF share={pct(t['uni_etf'], t['uni'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
