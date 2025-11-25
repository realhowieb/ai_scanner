

"""
Utilities for cleaning and normalizing ticker symbols across providers.

This module focuses on **Yahoo Finance** compatibility by default, since the
app fetches data primarily via yfinance. We preserve exchange suffixes like
".TO" (TSX), ".L" (LSE), etc., while normalizing class shares to the Yahoo
format (e.g., BRK.B -> BRK-B, BF.B -> BF-B).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import re

# --- Known Yahoo exchange suffixes we should preserve (dot notation) ---
# NOTE: Order matters (longer first) to avoid partial suffix matches.
_YF_DOT_SUFFIXES: Tuple[str, ...] = (
    ".WI",  # when-issued
    ".TO", ".V", ".CN",             # Canada
    ".L", ".IL",                    # UK / IOB
    ".HK",                           # Hong Kong
    ".AX", ".NZ",                   # Australia / New Zealand
    ".PA", ".BE", ".F", ".SW", ".MU", ".DU", ".SG", ".DE",  # France/DE/CH
    ".MI", ".BR", ".MC", ".AS", ".CO", ".OL", ".ST",          # EU
    ".SA",                           # Brazil
)

# Class share punctuation that should become "-" for Yahoo (e.g., BRK.B -> BRK-B)
_CLASS_SEP_PATTERN = re.compile(r"(?<=^[A-Z0-9]{1,10})[\._/ ](?=[A-Z]{1,2}$)")

# Characters allowed in final Yahoo-normalized tickers (plus dot for suffixes)
_ALLOWED_PATTERN = re.compile(r"^[A-Z0-9\-\.]+$")

# Common junk/test/placeholder symbols that we should drop early
_BLOCKLIST = {
    "ZAZZT", "ZBZZT", "ZCZZT", "ZJZZT", "ZWZZT", "ZXYZ-A",
    "FILE", "FERA", "FORL", "YHNAU", "YOSH",
}

# SPAC units/warrants indicators frequently not wanted by default
_SPAC_UNIT_SUFFIXES = ("U", "UN", "-U", "-UN", ".U", ".UN")
_SPAC_WARRANT_SUFFIXES = ("W", "WS", "-W", "-WS", ".W", ".WS")

# Map special known cases to Yahoo tickers
_SPECIAL_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


def _split_base_and_suffix(t: str) -> Tuple[str, str]:
    """Split a ticker into base + preserved Yahoo dot-suffix.

    Examples:
        "TD.TO" -> ("TD", ".TO")
        "AAPL"  -> ("AAPL", "")
    """
    for suf in _YF_DOT_SUFFIXES:
        if t.endswith(suf):
            return t[: -len(suf)], suf
    return t, ""


def _replace_class_separator_to_dash(base: str) -> str:
    # Convert class/share separators to '-' (BRK.B -> BRK-B)
    return _CLASS_SEP_PATTERN.sub("-", base)


def _collapse_dashes(s: str) -> str:
    return re.sub(r"-{2,}", "-", s).strip("-")


def normalize_ticker(raw: str, provider: str = "yahoo") -> str:
    """Normalize a single ticker string.

    Steps:
      - Strip, upper-case, remove leading '$'
      - Preserve Yahoo exchange suffixes (e.g., .TO, .L)
      - Convert class separators to '-' for Yahoo (BRK.B -> BRK-B)
      - Remove spaces/underscores/slashes in the base part
      - Collapse multiple dashes
    """
    if raw is None:
        return ""

    t = str(raw).strip().upper()
    if not t:
        return ""

    # Remove a leading '$' often found in social feeds
    if t.startswith("$"):
        t = t[1:]

    # Quick special mapping
    if t in _SPECIAL_MAP:
        t = _SPECIAL_MAP[t]

    # Preserve known exchange dot-suffixes; normalize the base portion
    base, suf = _split_base_and_suffix(t)

    # Convert other separators in base to '-' if they look like class parts
    base = _replace_class_separator_to_dash(base)

    # Replace spaces/underscores/slashes in base that slipped through
    base = base.replace(" ", "-").replace("_", "-").replace("/", "-")

    # Collapse repeated dashes
    base = _collapse_dashes(base)

    t = base + suf

    # Final allowlist check (letters, digits, dash, dot)
    if not _ALLOWED_PATTERN.match(t):
        # If it doesn't match allowed characters, return empty string to drop it
        return ""
    return t


def is_probably_delisted(t: str) -> bool:
    """Heuristic filters for obviously bad/test symbols."""
    if not t:
        return True
    if t in _BLOCKLIST:
        return True
    if "ZZZT" in t:  # Yahoo test placeholders
        return True
    return False


def is_spac_unit_or_warrant(t: str) -> bool:
    """Return True if the ticker looks like a SPAC unit or warrant."""
    if not t:
        return False
    # Check end-with patterns on the base (before dot suffix)
    base, _ = _split_base_and_suffix(t)
    return (
        base.endswith(_SPAC_UNIT_SUFFIXES) or base.endswith(_SPAC_WARRANT_SUFFIXES)
    )


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sanitize_ticker_list(
    symbols: Iterable[str],
    *,
    provider: str = "yahoo",
    drop_spac_units: bool = True,
    drop_blocklisted: bool = True,
    unique: bool = True,
) -> List[str]:
    """Normalize and filter a collection of tickers.

    Args:
        symbols: raw tickers (any strings)
        provider: normalization target ("yahoo" supported)
        drop_spac_units: remove SPAC units/warrants by default
        drop_blocklisted: drop known junk/test symbols
        unique: de-duplicate while preserving order

    Returns:
        Cleaned list of tickers.
    """
    cleaned: List[str] = []
    for raw in symbols or []:
        t = normalize_ticker(raw, provider=provider)
        if not t:
            continue
        if drop_blocklisted and is_probably_delisted(t):
            continue
        if drop_spac_units and is_spac_unit_or_warrant(t):
            continue
        cleaned.append(t)

    if unique:
        cleaned = dedupe_preserve_order(cleaned)
    return cleaned


__all__ = [
    "normalize_ticker",
    "sanitize_ticker_list",
    "is_probably_delisted",
    "is_spac_unit_or_warrant",
    "dedupe_preserve_order",
]