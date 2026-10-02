"""P1-44: multi-user load test for the Neon database paths.

Simulates N signed-in users at once, each repeating the database work of a page
load through the app's own functions: session lookup, user record, settings,
watchlists, scan history, alerts, and (sometimes) a watchlist add/remove. For
each concurrency level it reports p50/p95/max latency per operation, errors,
and the peak number of database connections.

Run it against a Neon BRANCH (a copy of the database), never production:

    LOADTEST_DATABASE_URL=postgresql://...  python scripts/neon_load_test.py \
        --users 10,25,50,100 --seconds 60

Options:
    --reconnect      new connection for every operation (AI_SCANNER_DB_POOL=0),
                     the worst case for connection churn
    --schema-every-call
                     re-run the ensure-schema DDL on every call
                     (AI_SCANNER_SCHEMA_ONCE=0), the behavior before P1-44
    --write-ratio    share of page loads that also add/remove a watchlist ticker

Synthetic users are named loadtest+NNN@example.invalid; their rows are removed at
the end (and at the start, in case a previous run was interrupted). Results are
written to artifacts/neon_load_test.json and, in GitHub Actions, to the job
summary. No credentials are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

USER_FMT = "loadtest+{:03d}@example.invalid"
USER_LIKE = "loadtest+%@example.invalid"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "NFLX", "PLTR"]


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _configure(args) -> str:
    url = (os.environ.get("LOADTEST_DATABASE_URL") or "").strip()
    if not url:
        sys.exit("LOADTEST_DATABASE_URL is not set (point it at a Neon branch, not production).")
    for name in ("DATABASE_URL", "NEON_DATABASE_URL"):
        other = (os.environ.get(name) or "").strip()
        if other and (other == url or _host(other) == _host(url)):
            sys.exit(f"LOADTEST_DATABASE_URL points at the same host as {name}; refusing to load-test production.")
    # The app reads these; set them before importing any db module.
    os.environ["NEON_DATABASE_URL"] = url
    os.environ.pop("DATABASE_URL", None)
    os.environ["AI_SCANNER_SQLITE_FALLBACK"] = "false"
    os.environ["AI_SCANNER_DB_POOL"] = "0" if args.reconnect else "1"
    os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0" if args.schema_every_call else "1"
    return url


def _direct(url):
    import psycopg

    return psycopg.connect(url, autocommit=True, connect_timeout=10)


def _cleanup(url) -> None:
    with _direct(url) as conn:
        for sql in (
            "DELETE FROM watchlist_items WHERE watchlist_id IN (SELECT id FROM watchlists WHERE user_id LIKE %s)",
            "DELETE FROM watchlists WHERE user_id LIKE %s",
            "DELETE FROM auth_sessions WHERE username LIKE %s",
            "DELETE FROM user_settings WHERE user_id LIKE %s",
            "DELETE FROM users WHERE username LIKE %s",
        ):
            try:
                conn.execute(sql, (USER_LIKE,))
            except Exception:  # table may not exist yet on a fresh branch
                pass


def _seed(url, count: int) -> dict:
    """Create the synthetic users and one session each; return {user: cookie}."""
    from db.engine import get_neon_conn
    from db.schema import ensure_neon_users_schema
    from db.watchlists import add_tickers_to_watchlist
    from ui.auth_sessions import ensure_auth_sessions_schema, session_hash

    conn = get_neon_conn()
    if conn is None:
        sys.exit("Could not connect to LOADTEST_DATABASE_URL.")
    ensure_neon_users_schema(conn)
    ensure_auth_sessions_schema(conn)
    conn.close()

    tokens = {}
    expires = datetime.now(timezone.utc) + timedelta(days=1)
    with _direct(url) as direct:
        for i in range(count):
            user = USER_FMT.format(i)
            direct.execute(
                "INSERT INTO users (username, full_name, password, tier) VALUES (%s, %s, %s, 'pro') "
                "ON CONFLICT (username) DO NOTHING",
                (user, f"Load Test {i}", "!"),  # "!" is not a valid hash, so nobody can sign in
            )
            token = secrets.token_urlsafe(32)
            direct.execute(
                "INSERT INTO auth_sessions (username, expires_at, session_hash) VALUES (%s, %s, %s)",
                (user, expires, session_hash(token)),
            )
            tokens[user] = token
    for user in tokens:
        add_tickers_to_watchlist(user, random.sample(TICKERS, 4))
    # Create every table a page load touches before the timed run: on a fresh
    # branch, concurrent first-time CREATE TABLEs race (production has them).
    from db.alerts import list_alerts
    from db.user_settings import get_user_settings

    list_alerts(USER_FMT.format(0))
    get_user_settings(USER_FMT.format(0))
    return tokens


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.times: dict[str, list[float]] = {}
        self.errors: dict[str, int] = {}
        self.error_types: dict[str, str] = {}
        self.pages = 0

    def run(self, op: str, fn, *args):
        start = time.perf_counter()
        ok = True
        try:
            result = fn(*args)
        except Exception as exc:  # the type only: messages can carry connection details
            ok, result = False, None
            with self.lock:
                self.error_types.setdefault(op, type(exc).__name__)
        elapsed = (time.perf_counter() - start) * 1000
        with self.lock:
            self.times.setdefault(op, []).append(elapsed)
            if not ok:
                self.errors[op] = self.errors.get(op, 0) + 1
        return ok, result


def _page_load(rec: Recorder, user: str, token: str, write_ratio: float) -> None:
    from db.alerts import list_alerts
    from db.runs import list_runs
    from db.user_settings import get_user_settings
    from db.users import get_user_by_username
    from db.watchlists import add_to_watchlist, get_user_watchlist, list_watchlists, remove_from_watchlist
    from ui.auth_sessions import get_username_for_session

    ok, who = rec.run("session_lookup", get_username_for_session, token)
    if ok and who != user:
        with rec.lock:  # a lookup that returns no user is a failed page load
            rec.errors["session_lookup"] = rec.errors.get("session_lookup", 0) + 1
    rec.run("user_record", get_user_by_username, user)
    rec.run("user_settings", get_user_settings, user)
    rec.run("watchlists", list_watchlists, user)
    rec.run("watchlist_tickers", get_user_watchlist, user)
    rec.run("scan_history", list_runs, 20, False, user)
    rec.run("alerts", list_alerts, user)
    if random.random() < write_ratio:
        ticker = random.choice(TICKERS)
        rec.run("watchlist_write", add_to_watchlist, user, ticker)
        rec.run("watchlist_write", remove_from_watchlist, user, ticker)
    with rec.lock:
        rec.pages += 1


def _sample_connections(url, stop: threading.Event, peak: list) -> None:
    try:
        conn = _direct(url)
    except Exception:
        return
    with conn:
        while not stop.is_set():
            try:
                n = conn.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                ).fetchone()[0]
                peak[0] = max(peak[0], int(n))
            except Exception:
                pass
            stop.wait(0.25)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def run_level(url, tokens: dict, users: int, seconds: float, write_ratio: float) -> dict:
    rec = Recorder()
    stop, peak = threading.Event(), [0]
    sampler = threading.Thread(target=_sample_connections, args=(url, stop, peak), daemon=True)
    sampler.start()
    names = list(tokens)[:users]
    deadline = time.monotonic() + seconds

    def worker(user):
        time.sleep(random.random())  # stagger the first page loads
        while time.monotonic() < deadline:
            _page_load(rec, user, tokens[user], write_ratio)
            time.sleep(random.uniform(0.5, 2.0))  # think time between page loads

    threads = [threading.Thread(target=worker, args=(u,), daemon=True) for u in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(seconds + 120)
    stop.set()
    sampler.join(2)

    ops = {}
    for op, values in sorted(rec.times.items()):
        ops[op] = {
            "count": len(values),
            "errors": rec.errors.get(op, 0),
            "p50_ms": round(statistics.median(values), 1),
            "p95_ms": round(_pct(values, 95), 1),
            "max_ms": round(max(values), 1),
            "error_type": rec.error_types.get(op, ""),
        }
    return {
        "users": users,
        "seconds": seconds,
        "page_loads": rec.pages,
        "errors": sum(rec.errors.values()),
        "peak_connections": peak[0],
        "operations": ops,
    }


def _markdown(results: list[dict], args, url: str) -> str:
    host = _host(url)
    lines = [
        "## Neon load test (P1-44)",
        "",
        f"Host: `{host}` ({'pooled endpoint' if '-pooler' in host else 'direct endpoint'}) · "
        f"connections: {'new per operation' if args.reconnect else 'warm per user'} · "
        f"schema setup: {'every call' if args.schema_every_call else 'once per process'} · "
        f"{args.seconds:.0f}s per level · write ratio {args.write_ratio}",
        "",
        "| Users | Page loads | Errors | Peak connections | Slowest op p95 (ms) | Session lookup p50/p95 (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        ops = r["operations"]
        slowest = max(ops.items(), key=lambda kv: kv[1]["p95_ms"]) if ops else ("-", {"p95_ms": 0})
        sl = ops.get("session_lookup", {"p50_ms": 0, "p95_ms": 0})
        lines.append(
            f"| {r['users']} | {r['page_loads']} | {r['errors']} | {r['peak_connections']} | "
            f"{slowest[1]['p95_ms']} ({slowest[0]}) | {sl['p50_ms']} / {sl['p95_ms']} |"
        )
    lines += ["", "Per operation at the highest level:", "", "| Operation | Count | Errors | p50 | p95 | max |", "|---|---:|---:|---:|---:|---:|"]
    if results:
        for op, s in results[-1]["operations"].items():
            lines.append(f"| {op} | {s['count']} | {s['errors']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", default="10,25,50", help="comma-separated concurrency levels")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--write-ratio", type=float, default=0.2)
    ap.add_argument("--reconnect", action="store_true")
    ap.add_argument("--schema-every-call", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "neon_load_test.json"))
    args = ap.parse_args(argv)
    levels = sorted({int(x) for x in args.users.split(",") if x.strip()})
    if not levels or levels[0] < 1 or levels[-1] > 500:
        ap.error("--users must be between 1 and 500")

    url = _configure(args)
    _cleanup(url)
    results = []
    try:
        tokens = _seed(url, levels[-1])
        for n in levels:
            print(f"level {n} users for {args.seconds:.0f}s ...", flush=True)
            results.append(run_level(url, tokens, n, args.seconds, args.write_ratio))
            r = results[-1]
            print(f"  page loads={r['page_loads']} errors={r['errors']} peak connections={r['peak_connections']}", flush=True)
    finally:
        _cleanup(url)

    report = _markdown(results, args, url)
    print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "host": _host(url), "results": results}, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report)
    return 1 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
