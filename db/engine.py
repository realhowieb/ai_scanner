from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import Engine
import typing as _t


metadata = MetaData()


def normalize_postgres_url(url: str) -> str:
    """
    Accept common Postgres URL shapes and normalize to a SQLAlchemy-friendly URL.

    - Adds scheme prefix if missing.
    - Converts `postgres://` to `postgresql://`.
    - Leaves already-correct `postgresql+psycopg://` as-is.
    - Leaves query params intact.
    """
    if not url:
        return url

    # If scheme is missing, assume postgresql://
    if "://" not in url:
        url = f"postgresql://{url}"

    # Parse and possibly adjust scheme
    parsed = urlparse(url)
    scheme = parsed.scheme

    # Normalize legacy "postgres" scheme to SQLAlchemy's "postgresql"
    if scheme == "postgres":
        scheme = "postgresql"

    # Rebuild URL with possibly-updated scheme
    normalized = parsed._replace(scheme=scheme)
    return urlunparse(normalized)


def _needs_ssl(url: str) -> bool:
    """Return True if the URL likely targets a cloud Postgres that requires SSL.
    We assume SSL is needed if no explicit `sslmode` query param is present.
    """
    if "postgresql" not in url:
        return False

    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query or ""))
    return "sslmode" not in qs


def _with_ssl_required(url: str) -> str:
    """Append `sslmode=require` to the URL if missing."""
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query or ""))
    if qs.get("sslmode"):
        return url  # already present

    qs["sslmode"] = "require"
    new_query = urlencode(qs)
    return urlunparse(parsed._replace(query=new_query))


# --- Engine helpers (lazy, schema-aware) ------------------------------------

_engine: Engine | None = None  # cached engine; created on first use


def get_engine(*, echo: bool = False) -> Engine:
    """Return a cached SQLAlchemy Engine, creating it from env settings on first use.

    Env vars respected:
      - DB_URL  : full Postgres URL (Neon/Supabase/etc.). If present, used.
      - DB_PATH : SQLite filename when no DB_URL is provided. Default: "scanner.sqlite".
    """
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.getenv("DB_URL")
    db_path = os.getenv("DB_PATH", "scanner.sqlite")
    _engine = build_engine(db_url, db_path, echo=echo)
    return _engine


def init_db(*, db_url: str | None = None, db_path: str | None = None, echo: bool = False) -> Engine:
    """Initialize the database and create tables registered on this module's `metadata`.

    This will:
      1) Build an Engine (from args or env),
      2) Import `.schema` to ensure all Table objects are attached to `metadata`,
      3) Call `metadata.create_all(engine)`.

    Returns the Engine for convenience.
    """
    # Resolve configuration (args override env)
    if db_url is None:
        db_url = os.getenv("DB_URL")
    if db_path is None:
        db_path = os.getenv("DB_PATH", "scanner.sqlite")

    eng = build_engine(db_url, db_path, echo=echo)

    # Import table definitions so they're registered on this `metadata` before create_all.
    # Do it inside the function to avoid import-time cycles.
    try:
        from . import schema  # noqa: F401
    except Exception:
        try:
            import schema  # type: ignore  # noqa: F401
        except Exception:
            # If schema can't be imported, create_all will be a no-op; callers may handle separately.
            pass

    metadata.create_all(eng)

    # Cache the engine for subsequent `get_engine()` calls.
    global _engine
    _engine = eng
    return eng


def build_engine(db_url: str | None, db_path: str, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy Engine for Postgres (Neon/Supabase/etc.) or SQLite.

    - If `db_url` is provided, we normalize it and try psycopg2 first, then
      fall back to the modern `psycopg` driver if psycopg2 isn't installed.
    - If no `db_url` is provided, we build a local SQLite engine.
    - For Streamlit/multithreaded use, SQLite is created with
      `check_same_thread=False`.
    - For cloud Postgres providers that require TLS, ensure `sslmode=require`
      unless already specified in the URL.
    """
    if db_url:
        url = normalize_postgres_url(db_url)
        if _needs_ssl(url):
            url = _with_ssl_required(url)

        try:
            # Try standard psycopg2 driver first
            return create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=1800,  # avoid stale connections on cloud DBs
                echo=echo,
            )
        except ModuleNotFoundError as e:
            if "psycopg2" in str(e):
                # Fallback to `psycopg` (the modern driver)
                fallback_url = url.replace("postgresql://", "postgresql+psycopg://", 1)
                return create_engine(
                    fallback_url,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    echo=echo,
                )
            # Re-raise any other unexpected exceptions
            raise
    else:
        # Ensure directory exists for SQLite file
        db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        return create_engine(
            f"sqlite:///{db_path}",
            echo=echo,
            connect_args={"check_same_thread": False},
        )

# Expose a module-level `engine` for code that imports it directly. If creation
# fails (e.g., missing driver), we leave it as None; callers can use `get_engine()` or `init_db()`.
try:
    engine = get_engine()
except Exception:  # pragma: no cover
    engine = None  # type: ignore[assignment]