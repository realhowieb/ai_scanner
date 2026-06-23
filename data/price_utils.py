"""Shared helpers for price provider code."""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence
import hashlib
import random
import time

import pandas as pd


def cache_key(symbol: str, cfg: Any) -> str:
    """Build a stable cache key for a given symbol and price fetch config."""
    return f"{str(symbol).upper()}|{cfg.period}|{cfg.interval}|{'1' if cfg.prepost else '0'}"


def chunks(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    n = max(1, int(n))
    return (list(seq[i : i + n]) for i in range(0, len(seq), n))


def backoff_sleep(base: float, attempt: int) -> None:
    jitter = random.uniform(0.25, 0.75)
    delay = max(0.05, base * (2 ** max(0, attempt - 1)) * jitter)
    time.sleep(delay)


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort normalization of an OHLCV DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    df = df.copy()

    try:
        col_map: dict[object, str] = {}
        for col in list(df.columns):
            name = str(col).strip()
            lower = name.lower().replace("_", " ")

            if lower in ("open", "o"):
                new = "Open"
            elif lower in ("high", "h"):
                new = "High"
            elif lower in ("low", "l"):
                new = "Low"
            elif lower in ("close", "c"):
                new = "Close"
            elif lower in ("adj close", "adjclose", "adjusted close"):
                new = "Adj Close"
            elif lower in ("volume", "vol", "v"):
                new = "Volume"
            else:
                new = name

            col_map[col] = new

        if col_map:
            df = df.rename(columns=col_map)
            try:
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()]
            except (AttributeError, TypeError, ValueError):
                pass
    except (AttributeError, TypeError, ValueError):
        pass

    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        if column not in df.columns:
            continue
        try:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            try:
                df[column] = df[column].astype("float32")
            except TypeError:
                pass
        except (TypeError, ValueError):
            continue

    if "Volume" in df.columns:
        try:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            try:
                df["Volume"] = df["Volume"].astype("int64")
            except TypeError:
                pass
        except (TypeError, ValueError):
            pass

    try:
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
    except (AttributeError, TypeError, ValueError):
        pass

    return df


def frame_fingerprint(df: pd.DataFrame) -> str | None:
    """Build a lightweight fingerprint for a price DataFrame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not cols:
        return None

    try:
        sample_head = df[cols].head(4)
        sample_tail = df[cols].tail(4)
        sample = pd.concat([sample_head, sample_tail], axis=0)
        payload = sample.to_csv(index=False).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()
    except (AttributeError, TypeError, ValueError):
        return None
