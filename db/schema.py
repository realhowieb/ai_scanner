# db/schema.py
def ensure_neon_runs_schema(conn):
    cur = conn.cursor()
    cur.execute("""
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
    """)
    conn.commit()
    cur.close()


def ensure_sqlite_runs_schema(conn):
    """Ensure the SQLite 'runs' table exists with the expected schema."""
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