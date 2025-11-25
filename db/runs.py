import os
from datetime import datetime, timezone

from sqlalchemy import Table, Column, Integer, String, DateTime, select
from sqlalchemy.exc import NoResultFound

from .engine import build_engine, metadata

# Lazily build an engine from env or fallback to local SQLite
#   DB_URL    -> full SQLAlchemy URL (e.g., postgres+psycopg2://...)
#   DB_PATH   -> used only when DB_URL is missing; defaults to scanner.sqlite
_engine = build_engine(os.getenv("DB_URL"), os.getenv("DB_PATH", "scanner.sqlite"))

# --- Schema ---------------------------------------------------------------
# Keep this table minimal for now; expand as needed
runs = Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)),
    Column("results", String, nullable=False),
)

# Ensure tables exist
metadata.create_all(_engine)


# --- API -----------------------------------------------------------------

def save_run(name: str, results: str) -> int:
    """Save a scan run and return the inserted id."""
    ts = datetime.now(timezone.utc)
    with _engine.begin() as conn:
        ins = runs.insert().values(name=name, timestamp=ts, results=results)
        res = conn.execute(ins)
        # SQLAlchemy 1.4+: inserted_primary_key returns a sequence
        return int(res.inserted_primary_key[0])


def list_runs():
    """Return list of {id, name, timestamp} mappings, newest first."""
    stmt = select(runs.c.id, runs.c.name, runs.c.timestamp).order_by(runs.c.timestamp.desc())
    with _engine.connect() as conn:
        result = conn.execute(stmt)
        # Return as list of dicts for Streamlit ease
        return [dict(row) for row in result.mappings().all()]


def load_run_results(run_id: int) -> str:
    """Fetch the results payload for a given run id."""
    stmt = select(runs.c.results).where(runs.c.id == run_id)
    with _engine.connect() as conn:
        row = conn.execute(stmt).fetchone()
        if row is None:
            raise NoResultFound(f"No run found with id {run_id}")
        mapping = row._mapping if hasattr(row, "_mapping") else {"results": row[0]}
        return mapping["results"]
