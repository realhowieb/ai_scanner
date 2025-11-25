import re
from typing import Iterable, List, Mapping, Any, Optional

# --- Normalization helpers -------------------------------------------------

_TAG_TESTLIKE = re.compile(
    r"^(?:Z[BCJW]ZZT|ZAZZT|ZXYZ-A|TEST|DUMMY|FILE|PLACEHOLDER|FERA|FORL)$",
    re.IGNORECASE,
)

# Obvious non-equities we should ignore for our scans
_BAD_PREFIXES = (
    "^",   # indices like ^GSPC
)

_BAD_SUFFIXES = (
    "=F",  # futures
    "=X",  # FX
)

# Warrants / Units / Rights patterns that often break Yahoo or are not
# part of the target universe
_WARRANT_UNIT_RIGHT_PAT = re.compile(
    r"(?i)(?:[-\.](?:W|WS|WS[A-Z]?|WT|U|UN|R|RT)\b|\bWRT\b|\bUNIT\b)"
)

# Common class/share suffixes we allow. We normalize `.` to `-` for Yahoo
# (e.g., BRK.B -> BRK-B, BF.B -> BF-B)
_CLASS_SUFFIX_PAT = re.compile(r"(?i)\.(?=[A-Z])")

_ALNUM_DASH_DOT = re.compile(r"^[A-Z0-9\-\.]+$")


def normalize_ticker(s: str) -> str:
    """Return a Yahoo-friendly, uppercase ticker without leading/trailing fluff.

    Rules:
    - strip spaces and leading `$`
    - uppercase
    - convert class dot to dash (BRK.B -> BRK-B)
    - collapse multiple dashes
    """
    if not s:
        return ""
    s = s.strip().lstrip("$")
    if not s:
        return ""
    s = s.upper()
    # Turn class-dot into dash for Yahoo style
    s = _CLASS_SUFFIX_PAT.sub("-", s)
    # Collapse duplicate dashes
    s = re.sub(r"-+", "-", s)
    return s


def _looks_like_unit_warrant_right(sym: str) -> bool:
    """Heuristics to spot units/warrants/rights we don't want to scan.

    Examples filtered: XYZ-W, XYZ-WS, XYZ.W, XYZ-U, XYZ-UN, XYZ-R, XYZ-RT
    """
    if not sym:
        return False
    return bool(_WARRANT_UNIT_RIGHT_PAT.search(sym))


def _is_problem_symbol(sym: str) -> bool:
    """Symbols that are either placeholders, test symbols or known-bad."""
    if not sym:
        return True

    if _TAG_TESTLIKE.match(sym):
        return True

    for p in _BAD_PREFIXES:
        if sym.startswith(p):
            return True

    for sfx in _BAD_SUFFIXES:
        if sym.endswith(sfx):
            return True

    # reject anything with obvious path/separator chars
    if any(c in sym for c in ("/", "\\", " ", ":")):
        return True

    # only allow alnum + dash + dot
    if not _ALNUM_DASH_DOT.match(sym):
        return True

    return False


# --- Public filters ---------------------------------------------------------

def sanitize_ticker_list(tickers: Iterable[str]) -> List[str]:
    """Normalize, de-dup and filter out problematic tickers.

    Returns a list preserving the original order of first occurrence.
    """
    seen = set()
    out: List[str] = []
    for raw in tickers or []:
        sym = normalize_ticker(str(raw))
        if not sym:
            continue
        if sym in seen:
            continue
        if _is_problem_symbol(sym):
            continue
        if _looks_like_unit_warrant_right(sym):
            continue
        seen.add(sym)
        out.append(sym)
    return out


def filter_problem_tickers(tickers: Iterable[str]) -> List[str]:
    """Remove known-bad tickers but *do not* normalize format.

    Use this when upstream expects original strings but you still want to
    drop junk (placeholders, indices, futures, warrants/units/rights).
    """
    out: List[str] = []
    for raw in tickers or []:
        sym = normalize_ticker(str(raw))
        if not sym:
            continue
        if _is_problem_symbol(sym):
            continue
        if _looks_like_unit_warrant_right(sym):
            continue
        out.append(sym)
    return out


def filter_us_tickers(tickers: Iterable[str]) -> List[str]:
    """Keep common US-style equities after normalization.

    Heuristics:
    - must pass problem filter
    - allow letters/digits plus optional class dash (e.g., BRK-B, BF-B)
    - drop obvious ADR suffix oddities like `.F` (already caught by _is_problem_symbol)
    """
    out: List[str] = []
    for raw in tickers or []:
        sym = normalize_ticker(str(raw))
        if not sym:
            continue
        if _is_problem_symbol(sym):
            continue
        if _looks_like_unit_warrant_right(sym):
            continue
        # basic US equity pattern: alnum with optional single dash block for class
        if re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)?", sym):
            out.append(sym)
    return out


# --- Price filter helper ----------------------------------------------------

def filter_tickers_by_price(
    tickers: Iterable[str],
    price_lookup: Mapping[str, Any],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[str]:
    """Return tickers whose latest *Close* price is within [min_price, max_price].

    This is a lightweight helper that doesn't require pandas imports here.

    Parameters
    ----------
    tickers : iterable of str
        Symbols to evaluate.
    price_lookup : Mapping[str, Any]
        One of the following:
        - dict[str, float] mapping symbol -> latest close price
        - dict[str, pandas.DataFrame] where each df has a 'Close' column
    min_price, max_price : float | None
        Inclusive bounds. If None, that side is ignored.

    Returns
    -------
    list[str]
        Filtered list preserving original order.
    """
    out: List[str] = []
    # Normalize bounds once
    lo = float(min_price) if min_price is not None else None
    hi = float(max_price) if max_price is not None else None

    for raw in tickers or []:
        sym = normalize_ticker(str(raw))
        if not sym:
            continue
        # grab price from mapping
        p = None
        if hasattr(price_lookup, 'get'):
            val = price_lookup.get(sym)
            if isinstance(val, (int, float)):
                p = float(val)
            else:
                # Duck-typed DataFrame: try `val['Close'].iloc[-1]`
                try:
                    close_series = getattr(val, '__getitem__')('Close')  # val['Close']
                    # support both pandas Series/DataFrame column
                    if hasattr(close_series, 'iloc'):
                        last_val = close_series.iloc[-1]
                    else:
                        last_val = close_series[-1]
                    p = float(last_val)
                except Exception:
                    p = None
        if p is None:
            continue
        if lo is not None and p < lo:
            continue
        if hi is not None and p > hi:
            continue
        out.append(sym)
    return out


def filter_by_dollar_volume(
    tickers: Iterable[str],
    data_lookup: Mapping[str, Any],
    min_dollar_volume: Optional[float] = None,
    max_dollar_volume: Optional[float] = None,
) -> List[str]:
    """Filter tickers by *dollar volume* using lightweight, duck-typed access.

    Accepts several shapes for `data_lookup` values:

    - Tuple/list: (price, volume)
    - Dict-like: {'price': ..., 'volume': ...} or {'close': ..., 'volume': ...}
    - DataFrame-like: df with 'Close' and 'Volume' columns (we read the last row)

    Notes
    -----
    - No hard dependency on pandas here; we just access attributes/keys if present.
    - If price or volume can't be retrieved or parsed, the symbol is skipped.
    - Bounds are inclusive.
    """
    out: List[str] = []
    lo = float(min_dollar_volume) if min_dollar_volume is not None else None
    hi = float(max_dollar_volume) if max_dollar_volume is not None else None

    for raw in tickers or []:
        sym = normalize_ticker(str(raw))
        if not sym:
            continue

        val = data_lookup.get(sym) if hasattr(data_lookup, "get") else None
        if val is None:
            continue

        price = None
        volume = None

        # Case 1: tuple/list (price, volume)
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            p, v = val[0], val[1]
            try:
                price = float(p)
                volume = float(v)
            except Exception:
                price = None
                volume = None

        # Case 2: dict-like with keys
        elif hasattr(val, "get"):
            # try common key spellings
            p_key = "price" if "price" in val else ("close" if "close" in val else None)
            v_key = "volume" if "volume" in val else None
            if p_key and v_key:
                try:
                    price = float(val[p_key])
                    volume = float(val[v_key])
                except Exception:
                    price = None
                    volume = None

        # Case 3: DataFrame-like (with 'Close' and 'Volume' columns)
        if price is None or volume is None:
            try:
                close_col = val["Close"] if hasattr(val, "__getitem__") else None
                vol_col = val["Volume"] if hasattr(val, "__getitem__") else None
                if close_col is not None and vol_col is not None:
                    # support objects with .iloc, else try indexing
                    last_close = close_col.iloc[-1] if hasattr(close_col, "iloc") else close_col[-1]
                    last_vol = vol_col.iloc[-1] if hasattr(vol_col, "iloc") else vol_col[-1]
                    price = float(last_close)
                    volume = float(last_vol)
            except Exception:
                price = None
                volume = None

        if price is None or volume is None:
            continue

        dv = price * volume
        if lo is not None and dv < lo:
            continue
        if hi is not None and dv > hi:
            continue
        out.append(sym)

    return out