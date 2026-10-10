# db/engine.py
import os
import sqlite3
from pathlib import Path

from db.traffic import connect_options, count

try:
    import streamlit as st
except Exception:  # pragma: no cover - depends on optional UI dependency
    class _StreamlitShim:
        secrets: dict = {}

        @staticmethod
        def caption(*_args, **_kwargs):
            return None

        @staticmethod
        def cache_data(*_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    st = _StreamlitShim()  # type: ignore[assignment]

DB_PATH = Path(__file__).resolve().parent.parent / "scanner.sqlite"

# ---------------------------------------------------------------------------
# Connection reuse: 40+ call sites open a Neon connection, use it once, and
# close it — each open is a full TCP+TLS+auth handshake (~200-500ms) to a
# remote Postgres. Keep one warm connection per thread behind a proxy whose
# close() only ends the transaction (rollback); checkout validates it with rollback() (one cheap round
# trip that also clears any dangling transaction) and reconnects when dead.
# Set AI_SCANNER_DB_POOL=0 to restore connect-per-call behavior.
# ---------------------------------------------------------------------------
import threading as _threading

_pool_local = _threading.local()


class _WarmConn:
    """Proxy over a psycopg connection that keeps it warm on close()."""

    __slots__ = ("_conn",)

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_conn"), name, value)

    def close(self):
        """Keep the socket warm but end the transaction, as a real close() would.
        A warm connection left idle in a transaction keeps its table locks until
        the thread's next checkout, so any ALTER TABLE elsewhere (a deploy, a
        job, another process's schema check) waits on it and every query queued
        behind that ALTER stalls (API acceptance run)."""
        try:
            object.__getattribute__(self, "_conn").rollback()
        except Exception:
            pass

    # Dunder methods bypass __getattr__ (looked up on the class), so the
    # context-manager protocol must be implemented explicitly. psycopg3's own
    # `with conn:` CLOSES the connection on exit; here we keep the transaction
    # semantics (commit on success, rollback on error) but keep the socket warm.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        conn = object.__getattribute__(self, "_conn")
        try:
            if exc_type is None:
                conn.commit()
            else:
                conn.rollback()
        except Exception:
            pass
        return False


def _pool_enabled() -> bool:
    return os.environ.get("AI_SCANNER_DB_POOL", "1").strip() != "0"


def _checkout_warm(url: str):
    """Return a warm per-thread connection, reconnecting when stale."""
    import psycopg

    conn = getattr(_pool_local, "conn", None)
    if conn is not None:
        try:
            conn.rollback()
            count(connections_reused=1)
            return _WarmConn(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _pool_local.conn = None
    real = psycopg.connect(url, row_factory=psycopg.rows.dict_row, connect_timeout=_connect_timeout(), **connect_options())
    count(connections_opened=1)
    _pool_local.conn = real
    return _WarmConn(real)


def release_thread_connection() -> None:
    """End any open transaction on this thread's warm connection (keeps the
    socket). For long-running servers whose helpers don't call close()."""
    conn = getattr(_pool_local, "conn", None)
    if conn is None:
        return
    try:
        conn.rollback()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        _pool_local.conn = None


def _connect_timeout() -> int:
    """libpq connect_timeout (seconds) so a network stall never hangs a page."""
    try:
        return max(1, int(os.environ.get("DB_CONNECT_TIMEOUT", "10")))
    except (TypeError, ValueError):
        return 10


# ---------------------------------------------------------------------------
# Schema setup once per process (P1-44). The ensure-schema helpers run
# CREATE TABLE / ALTER TABLE ... ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT
# EXISTS before their queries. Even when nothing changes, ALTER TABLE takes an
# ACCESS EXCLUSIVE lock and CREATE INDEX a SHARE lock, so with many users each
# request waited on every open write to the table and, while waiting, blocked
# every read queued behind it. The DDL is idempotent: run it the first time per
# process and database, then skip. Connections without psycopg connection info
# (SQLite, test fakes) always run it. AI_SCANNER_SCHEMA_ONCE=0 turns this off.
# ---------------------------------------------------------------------------
import functools as _functools

_schema_done: set = set()


def _database_key(conn):
    """(host, port, dbname) of a psycopg connection, else None."""
    try:
        info = conn.info
        key = (info.host, info.port, info.dbname)
    except Exception:
        return None
    if all(isinstance(part, (str, int)) for part in key):
        return key
    return None


def schema_once(fn):
    """Run an ensure-schema function ``fn(conn, ...)`` once per process and database."""

    @_functools.wraps(fn)
    def wrapper(conn, *args, **kwargs):
        key = _database_key(conn)
        if key is None or os.environ.get("AI_SCANNER_SCHEMA_ONCE", "1").strip() == "0":
            return fn(conn, *args, **kwargs)
        tag = (fn.__module__, fn.__qualname__, key)
        if tag in _schema_done:
            return None
        result = fn(conn, *args, **kwargs)  # an exception leaves it to run again next time
        _schema_done.add(tag)
        return result

    return wrapper


def get_neon_conn():
    """Return a new Neon PostgreSQL connection.

    We try, in order:
    - NEON_DATABASE_URL or DATABASE_URL env var
    - st.secrets["neon"]["database_url"]
    - st.secrets["DATABASE_URL"]
    - st.secrets["neon_database_url"]
    - st.secrets["database_url"]

    If no URL is configured, return None. If a URL is configured but
    the connection fails, also return None (callers should handle this).
    """
    url = None

    # 1) Environment variable (useful in many deployment setups)
    env_url = (
        os.environ.get("NEON_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("database_url")
    )
    if env_url:
        url = env_url

    # 2) Streamlit secrets nested key
    if url is None:
        try:
            url = st.secrets["neon"]["database_url"]  # type: ignore[index]
        except Exception:
            pass

    # 3) Streamlit secrets flat keys
    if url is None:
        for key in ("NEON_DATABASE_URL", "DATABASE_URL", "neon_database_url", "database_url"):
            try:
                candidate = st.secrets[key]  # type: ignore[index]
                if candidate:
                    url = candidate
                    break
            except Exception:
                continue

    if not url:
        # No Neon URL configured at all
        return None

    try:
        if _pool_enabled():
            return _checkout_warm(url)
        import psycopg

        # Bounded connect timeout so a network stall never hangs a page render
        # indefinitely (libpq connect_timeout, overridable via env). Run 30 P1.
        conn = psycopg.connect(url, row_factory=psycopg.rows.dict_row, connect_timeout=_connect_timeout(), **connect_options())
        count(connections_opened=1)
        return conn
    except ImportError:
        # Never echo the exception (could include the DSN / connection string).
        try:
            st.caption("⚠️ Database driver unavailable.")
        except Exception:
            pass
        return None
    except Exception:
        # Surface a gentle, SANITIZED hint — never the raw error (Run 30: the DSN
        # can appear in psycopg error text). Callers handle None.
        try:
            st.caption("⚠️ Database temporarily unavailable.")
        except Exception:
            pass
        return None

def get_sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@st.cache_data(show_spinner=False, ttl=60)
def get_db_status() -> str:
    """Return 'neon', 'sqlite', or 'none' based on actual connectivity."""
    # Try Neon
    try:
        conn = get_neon_conn()
        if conn is not None:
            conn.close()
            return "neon"
    except Exception:
        pass

    # Try SQLite
    try:
        conn = get_sqlite_conn()
        conn.close()
        return "sqlite"
    except Exception:
        return "none"
