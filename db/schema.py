# db/schema.py
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
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
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
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
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