"""Seed a throwaway Postgres for the pre-merge browser journey (.github/workflows/web.yml).

Creates two verified Pro test accounts and one full-market scan run so the web app has
something to show. Refuses to run unless DATABASE_URL points at localhost, so it can
never write to the live database.

    DATABASE_URL=postgresql://...@127.0.0.1/... HSF_E2E_PASSWORD_FILE=pw.txt python scripts/seed_e2e.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

USERS = ("e2e-pro@example.com", "e2e-other@example.com")
# (ticker, last, change %, gap %, breakout score, PreBreakout %, breakout?, EMA cross)
ROWS = [
    ("AAPL", 231.50, 3.1, 1.2, 18.0, 71.0, True, "golden"),
    ("MSFT", 512.20, 2.4, 0.8, 15.0, 64.0, True, "golden"),
    ("NVDA", 188.40, 4.6, 2.1, 21.0, 77.0, True, None),
    ("AMD", 164.10, 2.9, 1.5, 13.0, 58.0, False, "golden"),
    ("CROX", 92.75, 1.8, 0.4, 11.0, 52.0, True, None),
    ("ZS", 231.00, 2.2, 0.9, 12.0, 55.0, False, "golden"),
    ("TSEM", 61.30, 1.5, 0.3, 9.0, 47.0, True, None),
    ("RRC", 38.90, 1.1, 0.2, 8.0, 44.0, False, "golden"),
]


def main() -> int:
    url = os.environ.get("DATABASE_URL", "")
    if urlparse(url).hostname not in ("127.0.0.1", "localhost"):
        print("seed_e2e: DATABASE_URL must point at localhost; refusing to seed anything else.")
        return 2
    import bcrypt
    import pandas as pd

    from db.engine import get_neon_conn
    from db.runs import save_run
    from db.users import create_user_account

    password = Path(os.environ["HSF_E2E_PASSWORD_FILE"]).read_text(encoding="utf-8").strip()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(4)).decode()
    for email in USERS:
        try:
            create_user_account(email, hashed, tier="pro", full_name="E2E Tester")
        except ValueError:
            pass  # already seeded
    conn = get_neon_conn()
    cur = conn.cursor()
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE")
    cur.execute("UPDATE users SET email_verified = TRUE WHERE username = ANY(%s)", (list(USERS),))
    conn.commit()
    cur.close()

    frame = pd.DataFrame([
        {"Ticker": t, "Last": last, "PctChange": chg, "GapPct": gap, "BreakoutScore": score,
         "PreBreakoutProb%": prob, "IsBreakout": brk, "EMACross": ema, "Why": "Seeded for the browser journey",
         "Volume": 5_000_000, "RVOL": 1.8}
        for t, last, chg, gap, score, prob, brk, ema in ROWS])
    for _ in range(2):  # two runs, so "since the last scan" views have a previous one
        save_run(name="E2E seeded scan", label="US_MARKET", username="cron", row_count=len(frame),
                 duration_sec=1.0, results_json=frame.to_json(orient="records", date_format="iso"),
                 is_snapshot=False, allow_sqlite_fallback=False)
    print(f"seed_e2e: {len(USERS)} accounts and 2 scan runs of {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
