# db/user_settings.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .engine import get_db_status, get_neon_conn

logger = logging.getLogger(__name__)

# Enable verbose user_settings logging only when explicitly requested.
# Set DEBUG_USER_SETTINGS=1 in your environment to turn this on.
_DEBUG_USER_SETTINGS = os.getenv("DEBUG_USER_SETTINGS", "").strip().lower() in {"1", "true", "yes", "on"}


def _dbg(msg: str) -> None:
    if _DEBUG_USER_SETTINGS:
        logger.info(msg)


def _ensure_user_settings_schema() -> None:
    """
    Create user_settings table in Neon if it doesn't exist yet.
    Safe to call repeatedly; uses IF NOT EXISTS.
    """
    if get_db_status() != "neon":
        # Only supported on Neon; SQLite fallback users just won't persist settings
        return

    conn = get_neon_conn()
    if conn is None:
        return

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE,
                    universe TEXT,
                    min_price DOUBLE PRECISION,
                    max_price DOUBLE PRECISION,
                    min_dollar_vol DOUBLE PRECISION,
                    include_ta BOOLEAN,
                    apply_gap_filter BOOLEAN,
                    show_diagnostics_ui BOOLEAN,
                    min_gap DOUBLE PRECISION,
                    top_n INTEGER,
                    max_nasdaq_scan INTEGER,
                    max_combo_scan INTEGER,
                    premarket BOOLEAN,
                    afterhours BOOLEAN,
                    unusual_vol BOOLEAN,
                    enable_earnings_enrichment BOOLEAN,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            # Ensure new columns exist even if the table was created before they were added
            cur.execute(
                """
                ALTER TABLE user_settings
                    ADD COLUMN IF NOT EXISTS min_gap DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS top_n INTEGER,
                    ADD COLUMN IF NOT EXISTS max_nasdaq_scan INTEGER,
                    ADD COLUMN IF NOT EXISTS max_combo_scan INTEGER,
                    ADD COLUMN IF NOT EXISTS premarket BOOLEAN,
                    ADD COLUMN IF NOT EXISTS afterhours BOOLEAN,
                    ADD COLUMN IF NOT EXISTS unusual_vol BOOLEAN,
                    ADD COLUMN IF NOT EXISTS enable_earnings_enrichment BOOLEAN,
                    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_settings_user_id
                    ON user_settings(user_id);
                """
            )


def _get_conn():
    """
    Internal helper that returns a Neon connection with schema ensured,
    or None if Neon is not available.
    """
    if get_db_status() != "neon":
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    _ensure_user_settings_schema()
    return conn


def get_user_settings(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Load saved settings for a given user_id (username).
    Returns a dict with keys matching sidebar state, or None if not found.
    """
    conn = _get_conn()
    _dbg(f"[DEBUG user_settings] get_user_settings called with user_id={user_id!r}")
    _dbg(f"[DEBUG user_settings] db_status={get_db_status()!r}, conn_is_none={conn is None}")
    if conn is None:
        return None

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    universe,
                    min_price,
                    max_price,
                    min_dollar_vol,
                    include_ta,
                    apply_gap_filter,
                    show_diagnostics_ui,
                    min_gap,
                    top_n,
                    max_nasdaq_scan,
                    max_combo_scan,
                    premarket,
                    afterhours,
                    unusual_vol,
                    enable_earnings_enrichment
                FROM user_settings
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            _dbg(f"[DEBUG user_settings] primary lookup row for {user_id!r} = {row!r}")

            # Fallback: be tolerant of stray whitespace or case differences in user_id
            if not row:
                cur.execute(
                    """
                    SELECT
                        universe,
                        min_price,
                        max_price,
                        min_dollar_vol,
                        include_ta,
                        apply_gap_filter,
                        show_diagnostics_ui,
                        min_gap,
                        top_n,
                        max_nasdaq_scan,
                        max_combo_scan,
                        premarket,
                        afterhours,
                        unusual_vol,
                        enable_earnings_enrichment
                    FROM user_settings
                    WHERE TRIM(LOWER(user_id)) = TRIM(LOWER(%s))
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                _dbg(f"[DEBUG user_settings] fallback lookup row for {user_id!r} = {row!r}")

    if not row:
        return None

    # If the cursor returns a dict (e.g., RealDictCursor), we can return it directly.
    # This matches the structure expected by the rest of the app (keys: universe, min_price, etc.).
    if isinstance(row, dict):
        return {
            "universe": row.get("universe"),
            "min_price": float(row["min_price"]) if row.get("min_price") is not None else None,
            "max_price": float(row["max_price"]) if row.get("max_price") is not None else None,
            "min_dollar_vol": float(row["min_dollar_vol"]) if row.get("min_dollar_vol") is not None else None,
            "include_ta": bool(row["include_ta"]) if row.get("include_ta") is not None else None,
            "apply_gap_filter": bool(row["apply_gap_filter"]) if row.get("apply_gap_filter") is not None else None,
            "show_diagnostics_ui": bool(row["show_diagnostics_ui"]) if row.get("show_diagnostics_ui") is not None else None,
            "min_gap": float(row["min_gap"]) if row.get("min_gap") is not None else None,
            "top_n": int(row["top_n"]) if row.get("top_n") is not None else None,
            "max_nasdaq_scan": int(row["max_nasdaq_scan"]) if row.get("max_nasdaq_scan") is not None else None,
            "max_combo_scan": int(row["max_combo_scan"]) if row.get("max_combo_scan") is not None else None,
            "premarket": bool(row["premarket"]) if row.get("premarket") is not None else None,
            "afterhours": bool(row["afterhours"]) if row.get("afterhours") is not None else None,
            "unusual_vol": bool(row["unusual_vol"]) if row.get("unusual_vol") is not None else None,
            "enable_earnings_enrichment": bool(row["enable_earnings_enrichment"]) if row.get("enable_earnings_enrichment") is not None else None,
        }

    # Fallback: if we ever get a positional row (tuple), keep the original tuple-unpack logic.
    (
        universe,
        min_price,
        max_price,
        min_dollar_vol,
        include_ta,
        apply_gap_filter,
        show_diagnostics_ui,
        min_gap,
        top_n,
        max_nasdaq_scan,
        max_combo_scan,
        premarket,
        afterhours,
        unusual_vol,
        enable_earnings_enrichment,
    ) = row

    return {
        "universe": universe,
        "min_price": float(min_price) if min_price is not None else None,
        "max_price": float(max_price) if max_price is not None else None,
        "min_dollar_vol": float(min_dollar_vol) if min_dollar_vol is not None else None,
        "include_ta": bool(include_ta) if include_ta is not None else None,
        "apply_gap_filter": bool(apply_gap_filter) if apply_gap_filter is not None else None,
        "show_diagnostics_ui": bool(show_diagnostics_ui) if show_diagnostics_ui is not None else None,
        "min_gap": float(min_gap) if min_gap is not None else None,
        "top_n": int(top_n) if top_n is not None else None,
        "max_nasdaq_scan": int(max_nasdaq_scan) if max_nasdaq_scan is not None else None,
        "max_combo_scan": int(max_combo_scan) if max_combo_scan is not None else None,
        "premarket": bool(premarket) if premarket is not None else None,
        "afterhours": bool(afterhours) if afterhours is not None else None,
        "unusual_vol": bool(unusual_vol) if unusual_vol is not None else None,
        "enable_earnings_enrichment": bool(enable_earnings_enrichment) if enable_earnings_enrichment is not None else None,
    }


def upsert_user_settings(
    user_id: str,
    *,
    universe: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_dollar_vol: Optional[float] = None,
    include_ta: Optional[bool] = None,
    apply_gap_filter: Optional[bool] = None,
    show_diagnostics_ui: Optional[bool] = None,
    min_gap: Optional[float] = None,
    top_n: Optional[int] = None,
    max_nasdaq_scan: Optional[int] = None,
    max_combo_scan: Optional[int] = None,
    premarket: Optional[bool] = None,
    afterhours: Optional[bool] = None,
    unusual_vol: Optional[bool] = None,
    enable_earnings_enrichment: Optional[bool] = None,
) -> None:
    """
    Insert or update per-user scan settings in the Neon-backed user_settings table.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_settings (
                    user_id,
                    universe,
                    min_price,
                    max_price,
                    min_dollar_vol,
                    include_ta,
                    apply_gap_filter,
                    show_diagnostics_ui,
                    min_gap,
                    top_n,
                    max_nasdaq_scan,
                    max_combo_scan,
                    premarket,
                    afterhours,
                    unusual_vol,
                    enable_earnings_enrichment
                )
                VALUES (
                    %(user_id)s,
                    %(universe)s,
                    %(min_price)s,
                    %(max_price)s,
                    %(min_dollar_vol)s,
                    %(include_ta)s,
                    %(apply_gap_filter)s,
                    %(show_diagnostics_ui)s,
                    %(min_gap)s,
                    %(top_n)s,
                    %(max_nasdaq_scan)s,
                    %(max_combo_scan)s,
                    %(premarket)s,
                    %(afterhours)s,
                    %(unusual_vol)s,
                    %(enable_earnings_enrichment)s
                )
                ON CONFLICT (user_id) DO UPDATE
                SET
                    universe = EXCLUDED.universe,
                    min_price = EXCLUDED.min_price,
                    max_price = EXCLUDED.max_price,
                    min_dollar_vol = EXCLUDED.min_dollar_vol,
                    include_ta = EXCLUDED.include_ta,
                    apply_gap_filter = EXCLUDED.apply_gap_filter,
                    show_diagnostics_ui = EXCLUDED.show_diagnostics_ui,
                    min_gap = EXCLUDED.min_gap,
                    top_n = EXCLUDED.top_n,
                    max_nasdaq_scan = EXCLUDED.max_nasdaq_scan,
                    max_combo_scan = EXCLUDED.max_combo_scan,
                    premarket = EXCLUDED.premarket,
                    afterhours = EXCLUDED.afterhours,
                    unusual_vol = EXCLUDED.unusual_vol,
                    enable_earnings_enrichment = EXCLUDED.enable_earnings_enrichment;
                """,
                {
                    "user_id": user_id,
                    "universe": universe,
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_dollar_vol": min_dollar_vol,
                    "include_ta": include_ta,
                    "apply_gap_filter": apply_gap_filter,
                    "show_diagnostics_ui": show_diagnostics_ui,
                    "min_gap": min_gap,
                    "top_n": top_n,
                    "max_nasdaq_scan": max_nasdaq_scan,
                    "max_combo_scan": max_combo_scan,
                    "premarket": premarket,
                    "afterhours": afterhours,
                    "unusual_vol": unusual_vol,
                    "enable_earnings_enrichment": enable_earnings_enrichment,
                },
            )
        conn.commit()