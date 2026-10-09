"""Read-only production post-rescore audit. Run via Diagnostics workflow.

No schema initialization, freeze, maturation, model load/fit, or write helper
is called. The database itself enforces a read-only repeatable-read snapshot.
Only aggregate audit artifacts leave the process; no credentials/user data.
"""
from __future__ import annotations

import base64
import datetime as dt
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    if importlib.util.find_spec("sklearn") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "scikit-learn>=1.3,<2",
                        "-c", str(ROOT / "requirements.lock")], check=True)
    import psycopg
    from psycopg.rows import dict_row

    from analytics import ml_readiness as mr
    from analytics import post_rescore_audit as pa
    from analytics import research_dataset as rd
    from analytics.signal_outcomes import complete_session_bars, score_signal

    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("Production connection unavailable; refusing local/SQLite fallback")
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=15) as conn:
        conn.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        conn.execute("SET LOCAL statement_timeout = '30s'")
        now = conn.execute("SELECT now() AS now").fetchone()["now"]
        rows = conn.execute("""
            SELECT id, source, source_event_id, signal_type, ticker, fired_at, created_at,
                   setup_score, ai_confidence, prebreakout_prob, return_1d, return_3d, return_5d,
                   mfe_5d, mae_5d, outcome_computed_at, benchmark_symbol,
                   benchmark_return_1d, benchmark_return_3d, benchmark_return_5d, benchmark_computed_at,
                   jsonb_build_object('hsf_score',raw_signal->'hsf_score',
                     'score_version',raw_signal->'score_version','status',raw_signal->'status',
                     'primary_setup',raw_signal->'primary_setup',
                     'recommendation',raw_signal->'recommendation','direction',raw_signal->'direction',
                     'prebreakout_model_version',raw_signal->'prebreakout_model_version') AS raw_signal
            FROM signal_outcomes
            WHERE source='opportunity' AND fired_at >= '2026-09-01' AND fired_at < %s
            ORDER BY fired_at, id LIMIT 10001
        """, (pa.REPAIR_START,)).fetchall()
        if len(rows) > 10000:
            raise SystemExit("Audit row cap exceeded; refusing a truncated report")
        models = {}
        for table in ("prebreakout_models", "ai_confidence_models"):
            models[table] = conn.execute(
                f"SELECT id, model_version, trained_at, feature_names, metadata->>'target' AS target "
                f"FROM {table} WHERE is_active=true"  # nosec B608: fixed table allowlist
            ).fetchall()
        constraints = conn.execute("""SELECT contype, pg_get_constraintdef(oid) AS definition
                                      FROM pg_constraint WHERE conrelid='signal_outcomes'::regclass""").fetchall()
    # Release the snapshot before provider reads or local analysis.
    targets = [r for r in rows if mr.maturation_stage(r, now)["category"] in
               (mr.PREMATURE_LABEL_WRITE, mr.MISSING_PRICE_DATA)]
    probe = {}
    try:
        from data.price_alpaca import download_multi_alpaca

        tickers = sorted({r["ticker"] for r in targets})
        bars = download_multi_alpaca(tickers, period="90d", interval="1d", prepost=False, timeout_s=60.0) if tickers else {}
        for ticker in tickers:
            b = complete_session_bars((bars or {}).get(ticker), now)
            selected = [r for r in targets if r["ticker"] == ticker]
            closes = b["Close"].dropna() if b is not None and "Close" in b else []
            outcomes = [score_signal(b, r["fired_at"]) if b is not None else None for r in selected]
            probe[ticker] = {"observations": len(selected), "returned_bars": len(b) if b is not None else 0,
                             "valid_close_bars": len(closes), "first_bar": str(b.index.min()) if b is not None and len(b) else None,
                             "last_bar": str(b.index.max()) if b is not None and len(b) else None,
                             "entry_days": dict(Counter(str(rd.entry_day(r["fired_at"])) for r in selected)),
                             "scorable_now": sum(o is not None for o in outcomes),
                             "reason": "NO_RETURNED_BARS" if not len(closes) else "CANONICAL_SCORER_RETURNED_NONE" if all(o is None for o in outcomes) else "SCORABLE"}
    except Exception as e:
        probe = {"error_type": type(e).__name__, "status": "PROVIDER_PROBE_UNAVAILABLE"}
    result = pa.analyze(rows, models, now)
    result["database_constraints"] = constraints
    result["price_probe"] = probe
    result["repair_write_timestamps"] = dict(Counter(str(r["outcome_computed_at"]) for r in rows if pa.recovered(r)))
    result["repair_benchmark_timestamps"] = dict(Counter(str(r["benchmark_computed_at"]) for r in rows if pa.recovered(r)))
    result["git_sha"] = os.getenv("GITHUB_SHA")
    result["workflow_run_id"] = os.getenv("GITHUB_RUN_ID")
    result["repair_ids"] = sorted(r["id"] for r in rows if pa.recovered(r))
    text = json.dumps(result, default=str, allow_nan=False, indent=2)
    out = ROOT / "ml/reports/post_rescore_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print("POST_RESCORE_INTEGRITY", json.dumps(result["POST_RESCORE_INTEGRITY"]))
    print("DATASET_BEFORE", json.dumps(result["dataset_before"]))
    print("DATASET_AFTER", json.dumps(result["dataset_after"]))
    print("PRICE_PROBE", json.dumps(probe))
    print("PERFORMANCE", json.dumps(result["performance"]))
    print("=== POST_RESCORE_BUNDLE_BEGIN ===")
    print(base64.b64encode(gzip.compress(text.encode())).decode())
    print("=== POST_RESCORE_BUNDLE_END ===")
    return 0 if result["POST_RESCORE_INTEGRITY"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
