"""AI features for API clients (P1-71, Premium: can_ai_notes).

Same prompts, guardrails and limits as the web: every call goes through
ui.ai.ask_claude / ask_claude_chat (kill switch AI_ENABLED, per-account daily
cap AI_DAILY_LIMIT counted in db.ai_usage, request timeout, the Run 85
responsibility rules and the trade-instruction scrub). The web reads the
account from the Streamlit session; here it is passed explicitly.

Answers that are the same for everyone (the latest market scan's summary, a
ticker note, the brief narrative) are shared for 30 minutes per scan, so they
cost one call per scan, not one per user.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SHARED_TTL_S = 1800
MAX_CHAT_TURNS = 8          # ui.ai_chat._MAX_HISTORY
MAX_TURN_CHARS = 2000


class AIUnavailable(RuntimeError):
    """AI is switched off or not configured (503)."""


class AILimit(RuntimeError):
    """The account's daily AI limit is used up (429)."""


class AIFailed(RuntimeError):
    """The model call failed (502)."""


def _raise_for(err: Optional[str]) -> None:
    msg = err or "AI failed."
    low = msg.lower()
    if "usage limit" in low:
        raise AILimit(msg)
    if "not configured" in low or "disabled" in low or "requires the `anthropic` package" in low:
        raise AIUnavailable("AI is unavailable right now.")
    raise AIFailed("AI couldn't answer right now. Try again in a minute.")


def _ask(*, system: str, user: str, max_tokens: int, username: str, feature: str) -> str:
    from ui.ai import ask_claude

    text, err = ask_claude(system=system, user=user, max_tokens=max_tokens, username=username, feature=feature)
    if not text:
        _raise_for(err)
    return text


def _shared(key: Any, make) -> str:
    """One answer per key for SHARED_TTL_S; failures are not cached."""
    from api.today import _cached

    return _cached(("ai",) + tuple(key), make, ttl_s=SHARED_TTL_S)


def scan_df(username: str, run_id: Optional[int]):
    """(run_id, DataFrame): the latest market scan, or one of your saved scans."""
    from api.today import market_runs, run_df

    if run_id is None:
        runs = market_runs()
        if not runs:
            return None, None
        run_id = int(runs[0]["id"])
    else:
        from api.history import _owned_run

        if _owned_run(username, int(run_id)) is None:
            return None, None
    return int(run_id), run_df(int(run_id))


def summary(username: str, run_id: Optional[int], *, shared: bool) -> Dict[str, Any]:
    from ui.ai_summary import _MAX_ROWS, _SYSTEM_PROMPT, _build_table_text

    rid, df = scan_df(username, run_id)
    if df is None or len(df) == 0:
        return {"run_id": rid, "text": None}

    def make() -> str:
        return _ask(system=_SYSTEM_PROMPT,
                    user=(f"Here are the top {min(len(df), _MAX_ROWS)} scan results in HSF Score order (CSV):\n\n"
                          f"{_build_table_text(df)}\n\nExplain the leading setups in this order."),
                    max_tokens=1024, username=username, feature="scan_summary")

    text = _shared(("summary", rid), make) if shared else make()
    return {"run_id": rid, "text": text}


def ticker_note(username: str, ticker: str) -> Dict[str, Any]:
    """The web's per-result setup note, for a ticker in the latest market scan."""
    from ui.ai_summary import _SUMMARY_COLUMNS, _TICKER_SYSTEM_PROMPT

    rid, df = scan_df(username, None)
    if df is None or "Ticker" not in getattr(df, "columns", []):
        return {"run_id": rid, "ticker": ticker, "text": None}
    rows = df[df["Ticker"].astype(str).str.strip().str.upper() == ticker]
    if rows.empty:
        return {"run_id": rid, "ticker": ticker, "text": None}
    row = rows.iloc[0]
    metrics = "\n".join(f"- {c}: {row.get(c)}" for c in _SUMMARY_COLUMNS
                        if c in rows.columns and row.get(c) is not None and row.get(c) == row.get(c))

    def make() -> str:
        return _ask(system=_TICKER_SYSTEM_PROMPT,
                    user=f"Technical scan metrics for {ticker}:\n\n{metrics}\n\nExplain this setup.",
                    max_tokens=600, username=username, feature="ticker_deepdive")

    return {"run_id": rid, "ticker": ticker, "text": _shared(("note", rid, ticker), make)}


def chat(username: str, run_id: Optional[int], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Q&A about one scan. The client keeps the history (like the web's session),
    the server caps it at MAX_CHAT_TURNS and puts the scan in context first."""
    from ui.ai import ask_claude_chat
    from ui.ai_chat import _SYSTEM, _table

    rid, df = scan_df(username, run_id)
    if df is None or len(df) == 0:
        return {"run_id": rid, "answer": None}
    turns = [{"role": m["role"], "content": str(m["content"])[:MAX_TURN_CHARS]} for m in messages][-MAX_CHAT_TURNS * 2:]
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    api_messages = [{"role": "user", "content": f"Here are the current scan results (CSV):\n\n{_table(df)}"},
                    {"role": "assistant", "content": "Got it — I have the scan results. Ask away."}, *turns]
    answer, err = ask_claude_chat(system=_SYSTEM, messages=api_messages, max_tokens=500,
                                  username=username, feature="results_chat")
    if not answer:
        _raise_for(err)
    return {"run_id": rid, "answer": answer}


def brief_narrative(username: str) -> Dict[str, Any]:
    """The Market Brief's 2-3 sentence narrative, from real facts only."""
    from api.market import _brief_core
    from ui.market_brief import BRIEF_NARRATIVE_SYSTEM, _brief_narrative_facts

    core = _brief_core()
    data = (core or {}).get("data")
    facts = _brief_narrative_facts(data) if data else ""
    if not facts.strip():
        return {"snapshot_time": (data or {}).get("snapshot_time"), "text": None}
    ts = data.get("snapshot_time")
    text = _shared(("narrative", str(ts)), lambda: _ask(system=BRIEF_NARRATIVE_SYSTEM, user=f"Today's scan facts:\n{facts}",
                                                          max_tokens=200, username=username,
                                                          feature="market_brief_narrative"))
    return {"snapshot_time": ts, "text": text}
