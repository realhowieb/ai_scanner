"""Shared Claude (Anthropic) helper.

Centralizes API key handling, model selection, error mapping, and a single
chat entry point so every AI feature in the app is a thin wrapper. Best-effort:
callers get (text, error) and nothing raises.
"""
from __future__ import annotations

_MISCONFIGURED = "AI is not configured (missing ANTHROPIC_API_KEY)."


def is_configured() -> bool:
    """True if AI is enabled and an Anthropic API key is available."""
    try:
        from config import AI_ENABLED, ANTHROPIC_API_KEY
        return bool(AI_ENABLED and ANTHROPIC_API_KEY)
    except Exception:
        return False


def _over_daily_limit(username: str | None) -> bool:
    """True if this user has exceeded their per-day AI call budget."""
    if not username:
        return False
    try:
        from config import AI_DAILY_LIMIT
        if AI_DAILY_LIMIT <= 0:
            return False
        from db.ai_usage import ai_calls_today
        return ai_calls_today(username) >= AI_DAILY_LIMIT
    except Exception:
        return False  # fail-open: never block a user on a counting error


def ask_claude(
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
    model: str | None = None,
    username: str | None = None,
) -> tuple[str | None, str | None]:
    """Single-turn Claude call. Returns (text, error); never raises.

    Enforces the global kill switch, a per-request timeout, and (when a
    username is supplied) a per-user daily call cap.
    """
    try:
        from config import (
            AI_ENABLED,
            AI_REQUEST_TIMEOUT_SECONDS,
            ANTHROPIC_API_KEY,
            ANTHROPIC_MODEL,
        )
    except Exception:
        return None, "AI is not configured."

    if not AI_ENABLED:
        return None, "AI features are temporarily disabled."
    if not ANTHROPIC_API_KEY:
        return None, _MISCONFIGURED
    if _over_daily_limit(username):
        return None, "You've reached today's AI usage limit. Try again tomorrow."

    try:
        import anthropic
    except ImportError:
        return None, "AI requires the `anthropic` package (add to requirements)."

    try:
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=AI_REQUEST_TIMEOUT_SECONDS,
        )
        resp = client.messages.create(
            model=model or ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text and username:
            try:
                from db.ai_usage import record_ai_call
                record_ai_call(username)
            except Exception:
                pass
        return (text or None), (None if text else "Empty response from AI.")
    except anthropic.AuthenticationError:
        return None, "AI failed: invalid ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "AI rate-limited. Try again in a moment."
    except anthropic.APITimeoutError:
        return None, "AI request timed out. Try again."
    except anthropic.APIError as e:
        return None, f"AI failed: {e}"
    except Exception as e:
        return None, f"AI error: {type(e).__name__}: {e}"
