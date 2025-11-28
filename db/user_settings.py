# db/user_settings.py
from __future__ import annotations

from typing import Optional, Dict, Any

from .engine import get_neon_conn, get_db_status


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
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
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
                    show_diagnostics_ui
                FROM user_settings
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    (
        universe,
        min_price,
        max_price,
        min_dollar_vol,
        include_ta,
        apply_gap_filter,
        show_diagnostics_ui,
    ) = row

    return {
        "universe": universe,
        "min_price": float(min_price) if min_price is not None else None,
        "max_price": float(max_price) if max_price is not None else None,
        "min_dollar_vol": float(min_dollar_vol) if min_dollar_vol is not None else None,
        "include_ta": bool(include_ta) if include_ta is not None else None,
        "apply_gap_filter": bool(apply_gap_filter) if apply_gap_filter is not None else None,
        "show_diagnostics_ui": bool(show_diagnostics_ui) if show_diagnostics_ui is not None else None,
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
) -> None:
    """
    Insert or update a user's default scan settings.

    Called from the sidebar "Save as my default settings" button.
    """
    conn = _get_conn()
    if conn is None:
        raise RuntimeError("Neon DB not available for user settings")

    with conn:
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
                    show_diagnostics_ui
                )
                VALUES (
                    %(user_id)s,
                    %(universe)s,
                    %(min_price)s,
                    %(max_price)s,
                    %(min_dollar_vol)s,
                    %(include_ta)s,
                    %(apply_gap_filter)s,
                    %(show_diagnostics_ui)s
                )
                ON CONFLICT (user_id) DO UPDATE
                SET
                    universe = EXCLUDED.universe,
                    min_price = EXCLUDED.min_price,
                    max_price = EXCLUDED.max_price,
                    min_dollar_vol = EXCLUDED.min_dollar_vol,
                    include_ta = EXCLUDED.include_ta,
                    apply_gap_filter = EXCLUDED.apply_gap_filter,
                    show_diagnostics_ui = EXCLUDED.show_diagnostics_ui;
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
                },
            )