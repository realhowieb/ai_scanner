"""Provider failure diagnostics shared by scanner fetch paths.

The scanner should distinguish "no symbols matched filters" from "the market
data provider failed." This module is intentionally dependency-free so tests,
CI, and headless jobs can import it even when pandas/yfinance/Streamlit are not
installed yet.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProviderSummary:
    requested: int
    returned: int
    skipped: int
    categories: dict[str, int]
    message: str
    severe: bool = False


def classify_skip_reason(reason: object) -> str:
    """Map low-level fetch skip reasons to user-actionable categories."""
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"
    if "yf_not_installed" in text or "yfinance is not available" in text:
        return "provider_missing"
    if "rate" in text or "too many requests" in text or "429" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "unauthorized" in text or "forbidden" in text or "401" in text or "403" in text:
        return "auth"
    if "empty" in text or "missing_final" in text or "missing_after_rescue" in text:
        return "empty_response"
    if "duplicate_frame_like" in text:
        return "duplicate_data"
    if "error_download" in text or "batch_error" in text or "stream_error" in text:
        return "download_error"
    if "invalid_frame" in text or "error_normalize" in text:
        return "invalid_data"
    return "other"


def summarize_provider_skips(
    requested: int,
    returned: int,
    skipped: Iterable[object] | None,
) -> ProviderSummary:
    """Build a compact summary for UI captions, logs, and tests."""
    normalized = list(skipped or [])
    categories: Counter[str] = Counter()
    for item in normalized:
        reason = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else item
        categories[classify_skip_reason(reason)] += 1

    skipped_count = len(normalized)
    message = build_provider_message(requested, returned, categories)
    severe = bool(requested and returned == 0 and categories)
    return ProviderSummary(
        requested=int(requested or 0),
        returned=int(returned or 0),
        skipped=skipped_count,
        categories=dict(categories),
        message=message,
        severe=severe,
    )


def build_provider_message(
    requested: int,
    returned: int,
    categories: dict[str, int] | Counter[str],
) -> str:
    """Return a short, stable message describing provider health."""
    if requested <= 0:
        return "No tickers were requested."
    if returned <= 0:
        if categories.get("provider_missing"):
            return "No price data returned because yfinance is not available."
        if categories.get("rate_limited"):
            return "No price data returned; the provider appears rate-limited."
        if categories.get("timeout"):
            return "No price data returned; provider requests timed out."
        if categories.get("auth"):
            return "No price data returned; provider credentials were rejected."
        if categories.get("empty_response"):
            return "No price data returned; the provider returned empty responses."
        return "No price data returned from the configured providers."

    missing = max(0, requested - returned)
    if not missing:
        return f"Price data loaded for all {returned} requested symbols."

    parts = [
        f"{name.replace('_', ' ')}={count}"
        for name, count in sorted(categories.items())
        if count
    ]
    detail = ", ".join(parts[:4])
    suffix = f" ({detail})" if detail else ""
    return f"Price data loaded for {returned}/{requested} symbols; {missing} missing{suffix}."


def format_skip_examples(skipped: Sequence[object] | None, limit: int = 5) -> str:
    """Return a compact sample of skipped symbols for diagnostics."""
    examples: list[str] = []
    for item in list(skipped or [])[: max(0, int(limit))]:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            examples.append(f"{item[0]}: {item[1]}")
        else:
            examples.append(str(item))
    return "; ".join(examples)
