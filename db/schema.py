

"""Database schema for AI Scanner.

This module centralizes all SQLAlchemy Table definitions and a helper to
create the schema if it doesn't exist. Keep this module *pure* (no Streamlit),
so it can be safely imported from both UI and headless scheduler contexts.
"""
from __future__ import annotations

from sqlalchemy import (
    Table,
    Column,
    MetaData,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    BigInteger,
    Numeric,
    Index,
)
from sqlalchemy.sql import func

# Single, shared metadata for the whole app
metadata = MetaData()

# -----------------------------
# runs: one row per scan execution
# -----------------------------
# Notes:
# - "kind" captures the run type (e.g., "sp500", "nasdaq", "premarket", "postmarket").
# - "scheduled" indicates if it was triggered by the scheduler vs UI manual run.
# - "started_at"/"ended_at" are timezone-aware timestamps.
# - "elapsed_s" keeps the measured duration in seconds.
# - "universe_before"/"universe_after" record symbol universe sizes.
# - "status" is free-form (e.g., "ok", "error").
# - "error" stores any exception/trace captured.
# These columns reflect fields referenced throughout the app history.

runs = Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(120), nullable=False),
    Column("kind", String(40), nullable=False, index=True),
    Column("scheduled", Boolean, nullable=False, server_default="0"),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("ended_at", DateTime(timezone=True)),
    Column("elapsed_s", Float),
    Column("tickers_count", Integer),
    Column("universe_before", Integer),
    Column("universe_after", Integer),
    Column("notes", Text),
    Column("status", String(20), index=True),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
)

Index("ix_runs_started_at", runs.c.started_at)
Index("ix_runs_kind_started", runs.c.kind, runs.c.started_at.desc())

# --------------------------------
# run_results: rows produced by a run
# --------------------------------
# Stores scan outputs (e.g., breakout/gap/volume metrics). Keep names stable
# with prior code references, so saving won't fail on missing columns.

run_results = Table(
    "run_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("runs.id", ondelete="CASCADE"), index=True, nullable=False),
    Column("symbol", String(24), index=True, nullable=False),
    Column("price", Numeric(18, 6)),
    Column("change_abs", Numeric(18, 6)),
    Column("change_pct", Float),
    Column("breakout_pct", Float),
    Column("volume", BigInteger),
    Column("rel_vol", Float),
    Column("rsi", Float),
    Column("sma20", Float),
    Column("sma50", Float),
    Column("sma200", Float),
    Column("klass", String(24)),  # classification label (e.g., A/B/C); use "klass" to avoid Python keyword overlap
    Column("tags", String(200)),
    Column("source", String(40)),  # which scanner produced this row (e.g., breakout, premarket, postmarket)
    # Pre/Post-market optional fields
    Column("premarket_price", Numeric(18, 6)),
    Column("premarket_change_abs", Numeric(18, 6)),
    Column("premarket_change_pct", Float),
    Column("postmarket_price", Numeric(18, 6)),
    Column("postmarket_change_abs", Numeric(18, 6)),
    Column("postmarket_change_pct", Float),
    # Fundamentals/metadata (optional, if available)
    Column("sector", String(80)),
    Column("industry", String(120)),
    Column("ts", DateTime(timezone=True), server_default=func.now(), index=True),
)

Index("ix_results_symbol_run", run_results.c.symbol, run_results.c.run_id)

# -----------------------------
# Helper to create all tables
# -----------------------------

def create_all(engine) -> None:
    """Create all tables if they don't already exist.

    Usage (once at app start):
        from .engine import engine
        from .schema import create_all
        create_all(engine)
    """
    metadata.create_all(engine)