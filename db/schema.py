# db/schema.py
def ensure_neon_scan_errors_schema(conn):
    """Ensure the scan_errors table exists for telemetry logging."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_errors (
            id SERIAL PRIMARY KEY,
            context TEXT,
            username TEXT,
            universe TEXT,
            ticker_count INTEGER,
            error_type TEXT,
            message TEXT,
            traceback TEXT,
            occurred_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_scan_errors_occurred ON scan_errors (occurred_at DESC)"
    )
    conn.commit()
    cur.close()


def ensure_neon_runs_schema(conn):
    """Ensure the Neon 'runs' table exists with the expected schema, including watchlist_id."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            results_json TEXT NOT NULL,
            label TEXT,
            username TEXT,
            row_count INTEGER,
            duration_sec DOUBLE PRECISION,
            is_snapshot BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Add watchlist_id if it does not exist yet (Postgres/Neon supports IF NOT EXISTS here)
    cur.execute(
        """
        ALTER TABLE runs
        ADD COLUMN IF NOT EXISTS watchlist_id BIGINT
        """
    )
    conn.commit()
    cur.close()


def ensure_sqlite_runs_schema(conn):
    """Ensure the SQLite 'runs' table exists with the expected schema, including watchlist_id."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            results_json TEXT NOT NULL,
            label TEXT,
            username TEXT,
            row_count INTEGER,
            duration_sec REAL,
            is_snapshot INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # SQLite does not support IF NOT EXISTS on ADD COLUMN, so guard with PRAGMA
    cur.execute("PRAGMA table_info(runs)")
    cols = [row[1] for row in cur.fetchall()]
    if "watchlist_id" not in cols:
        cur.execute("ALTER TABLE runs ADD COLUMN watchlist_id INTEGER")
    conn.commit()
    cur.close()


def ensure_neon_users_schema(conn):
    """Ensure the Neon 'users' table exists with the expected schema."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password TEXT NOT NULL,
            tier TEXT DEFAULT 'basic',
            is_admin BOOLEAN DEFAULT FALSE,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            stripe_price_id TEXT,
            plan_updated_at TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_price_id TEXT")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_updated_at TIMESTAMPTZ")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE")
    conn.commit()
    cur.close()


def ensure_neon_login_attempts_schema(conn):
    """Ensure the login_attempts table exists for rate limiting."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS login_attempts (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            attempted_at TIMESTAMPTZ DEFAULT NOW(),
            success BOOLEAN DEFAULT FALSE,
            ip_address TEXT,
            failure_reason TEXT
        )
        """
    )
    cur.execute("ALTER TABLE login_attempts ADD COLUMN IF NOT EXISTS failure_reason TEXT")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_login_attempts_username_time ON login_attempts (username, attempted_at)"
    )
    conn.commit()
    cur.close()


def ensure_sqlite_users_schema(conn):
    """Ensure the SQLite 'users' table exists with the expected schema."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password TEXT NOT NULL,
            tier TEXT DEFAULT 'basic',
            is_admin INTEGER DEFAULT 0,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            stripe_price_id TEXT,
            plan_updated_at TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute("PRAGMA table_info(users)")
    cols = {row[1] for row in cur.fetchall()}
    if "stripe_customer_id" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN stripe_customer_id TEXT")
    if "stripe_subscription_id" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT")
    if "stripe_price_id" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN stripe_price_id TEXT")
    if "plan_updated_at" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN plan_updated_at TIMESTAMP")
    if "is_admin" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    conn.commit()
    cur.close()


def ensure_sqlite_login_attempts_schema(conn):
    """Ensure the login_attempts table exists for rate limiting (SQLite)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success INTEGER DEFAULT 0,
            ip_address TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_login_attempts_username_time ON login_attempts (username, attempted_at)"
    )
    conn.commit()
    cur.close()


def ensure_neon_watchlists_schema(conn):
    """Ensure the Neon 'watchlists' and 'watchlist_items' tables exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlists (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist_items (
            id BIGSERIAL PRIMARY KEY,
            watchlist_id BIGINT NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
            ticker TEXT NOT NULL
        )
        """
    )
    conn.commit()
    cur.close()


def ensure_sqlite_watchlists_schema(conn):
    """Ensure the SQLite 'watchlists' and 'watchlist_items' tables exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watchlist_id INTEGER NOT NULL,
            ticker TEXT NOT NULL
        )
        """
    )
    conn.commit()
    cur.close()
