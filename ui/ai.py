"""Shared Claude (Anthropic) helper.

Centralizes API key handling, model selection, error mapping, and a single
chat entry point so every AI feature in the app is a thin wrapper. Best-effort:
callers get (text, error) and nothing raises.
"""
from __future__ import annotations

_MISCONFIGURED = "AI is not configured (missing ANTHROPIC_API_KEY)."


def is_configured() -> bool:
    """True if an Anthropic API key is available."""
    try:
        from config import ANTHROPIC_API_KEY
        return bool(ANTHROPIC_API_KEY)
    except Exception:
        return False


def ask_claude(
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    """Single-turn Claude call. Returns (text, error); never raises."""
    try:
        from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    except Exception:
        return None, "AI is not configured."

    if not ANTHROPIC_API_KEY:
        return None, _MISCONFIGURED

    try:
        import anthropic
    except ImportError:
        return None, "AI requires the `anthropic` package (add to requirements)."

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=model or ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return (text or None), (None if text else "Empty response from AI.")
    except anthropic.AuthenticationError:
        return None, "AI failed: invalid ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "AI rate-limited. Try again in a moment."
    except anthropic.APIError as e:
        return None, f"AI failed: {e}"
    except Exception as e:
        return None, f"AI error: {type(e).__name__}: {e}"
