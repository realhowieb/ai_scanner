from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def reeval_inactive_symbols(
    conn,
    *,
    universe: str = "nasdaq",
    limit: int = 2000,
    chunk_size: int = 50,
    min_rows_required: int = 10,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Re-evaluate inactive symbols safely.

    Phase 1: Structural pass/fail using ticker regex.
    Phase 2: Price probe remaining inactive candidates in chunks.
    """

    # ---------- Phase 1: structural gating ----------
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE universe_symbols
            SET
              is_active = TRUE,
              inactive_reason = NULL,
              last_checked_at = NOW(),
              last_check_status = 'structural_pass',
              check_fail_count = 0
            WHERE universe = %s
              AND (ticker ~ '^[A-Z]{1,5}$');
            """,
            (universe,),
        )
        structural_activated = cur.rowcount

        cur.execute(
            """
            UPDATE universe_symbols
            SET
              is_active = FALSE,
              inactive_reason = COALESCE(inactive_reason, 'structural_fail'),
              last_checked_at = NOW(),
              last_check_status = 'structural_fail'
            WHERE universe = %s
              AND NOT (ticker ~ '^[A-Z]{1,5}$');
            """,
            (universe,),
        )
        structural_deactivated = cur.rowcount

    conn.commit()

    # ---------- Phase 2: price-probe remaining inactive ----------
    # Pick only symbols that are structurally valid but still inactive
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ticker
            FROM universe_symbols
            WHERE universe = %s
              AND is_active = FALSE
              AND (ticker ~ '^[A-Z]{1,5}$')
            ORDER BY ticker
            LIMIT %s;
            """,
            (universe, limit),
        )
        candidates = [r[0] for r in cur.fetchall()]

    total = len(candidates)
    if total == 0:
        return {
            "structural_activated": structural_activated,
            "structural_deactivated": structural_deactivated,
            "probed": 0,
            "activated": 0,
            "still_inactive": 0,
        }

    # Use your existing price fetcher if you have it (preferred)
    # This keeps behavior consistent with your scanner.
    try:
        from data.prices import fetch_price_data_batch  # type: ignore
    except Exception:
        fetch_price_data_batch = None  # type: ignore

    activated: list[str] = []
    failed: list[str] = []

    processed = 0
    for chunk in _chunked(candidates, chunk_size):
        processed += len(chunk)

        if progress_cb:
            progress_cb(processed, total, f"Probing {chunk[0]}..{chunk[-1]}")

        ok_symbols: set[str] = set()

        try:
            if fetch_price_data_batch is not None:
                price_map, _skipped = fetch_price_data_batch(chunk)
                for sym, df in (price_map or {}).items():
                    if df is not None and len(df) >= min_rows_required:
                        ok_symbols.add(sym)
            else:
                # Fallback (only if you don’t have the shared fetcher)
                import yfinance as yf
                df = yf.download(
                    tickers=" ".join(chunk),
                    period="60d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                # df may be multiindex if multiple tickers.
                # We'll just consider a symbol OK if we can extract any non-empty series.
                for sym in chunk:
                    try:
                        sub = df[sym] if hasattr(df, "__getitem__") else None
                        if sub is not None and len(sub) >= min_rows_required:
                            ok_symbols.add(sym)
                    except Exception:
                        pass
        except Exception:
            ok_symbols = set()

        for sym in chunk:
            if sym in ok_symbols:
                activated.append(sym)
            else:
                failed.append(sym)

        # Persist chunk results
        with conn.cursor() as cur:
            if activated:
                # only update those in this chunk
                to_activate = [s for s in chunk if s in ok_symbols]
                if to_activate:
                    cur.execute(
                        """
                        UPDATE universe_symbols
                        SET
                          is_active = TRUE,
                          inactive_reason = NULL,
                          last_checked_at = NOW(),
                          last_check_status = 'price_ok',
                          check_fail_count = 0
                        WHERE universe = %s
                          AND ticker = ANY(%s);
                        """,
                        (universe, to_activate),
                    )

            if failed:
                to_fail = [s for s in chunk if s not in ok_symbols]
                if to_fail:
                    cur.execute(
                        """
                        UPDATE universe_symbols
                        SET
                          is_active = FALSE,
                          inactive_reason = COALESCE(inactive_reason, 'no_price'),
                          last_checked_at = NOW(),
                          last_check_status = 'no_price',
                          check_fail_count = COALESCE(check_fail_count, 0) + 1
                        WHERE universe = %s
                          AND ticker = ANY(%s);
                        """,
                        (universe, to_fail),
                    )

        conn.commit()

    return {
        "structural_activated": structural_activated,
        "structural_deactivated": structural_deactivated,
        "probed": total,
        "activated": len(set(activated)),
        "still_inactive": len(set(failed)),
    }