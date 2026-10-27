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

# Appended to a prompt that carries Outcome Intelligence evidence (api.outcomes.ai_evidence).
EVIDENCE_RULES = (
    "\n\nHistorical HSF evidence may follow the metrics. If you mention it, state the matured sample size "
    "and the comparison with SPY, say plainly when the evidence label is INSUFFICIENT or LIMITED, and treat it "
    "as a description of past signals only. Never present it as a prediction, probability of profit, or "
    "guarantee, and never single out the best horizon or period."
)


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
    """Claude's per-result research note, for a ticker in the latest market scan."""
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

    from api.outcomes import ai_evidence

    evidence = ai_evidence(ticker)  # full matured history, every horizon, sample sizes; None when none

    def make() -> str:
        if not evidence:
            return _ask(system=_TICKER_SYSTEM_PROMPT,
                        user=f"Technical scan metrics for {ticker}:\n\n{metrics}\n\nExplain this setup.",
                        max_tokens=600, username=username, feature="ticker_deepdive")
        return _ask(system=_TICKER_SYSTEM_PROMPT + EVIDENCE_RULES,
                    user=(f"Technical scan metrics for {ticker}:\n\n{metrics}\n\n{evidence}\n\n"
                          "Explain this setup."),
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
    api_messages = [{"role": "user", "content": f"Here are the current scan results in HSF Score order (CSV):\n\n{_table(df)}"},
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


WATCHLIST_NOTE_SYSTEM = (
    "You write a short morning rundown of one trader's watchlist, from HSF scan facts only. "
    "In 3-6 plain sentences, say what changed since the previous scan: which stocks gained or lost "
    "HSF Score or rank, which became or stopped being ranked setups, unusual volume, and earnings coming up. "
    "Lead with the biggest changes and group quiet names together in one clause. Use ONLY the facts "
    "provided: do NOT invent prices, news, levels, forecasts or tickers. Describe conditions and what to "
    "watch; never predict, never tell the reader to buy, sell or hold. No emojis, no headings, no preamble."
)
WATCHLIST_NOTE_MAX_SYMBOLS = 40


def _fmt(v: Any, nd: int = 0, suffix: str = "") -> Optional[str]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f"{f:.{nd}f}{suffix}"


def watchlist_facts(intel: Dict[str, Any], earnings: Dict[str, Dict[str, Any]]) -> str:
    """One line of facts per symbol from GET /v1/watchlists/{id}/intelligence, biggest
    score moves first. Empty when the list has no symbols or no scan is available."""
    items = list(intel.get("items") or [])
    if not items or not intel.get("scan_available"):
        return ""
    items.sort(key=lambda i: -abs(float(i.get("score_change") or 0)))
    lines = []
    for i in items[:WATCHLIST_NOTE_MAX_SYMBOLS]:
        t = str(i.get("ticker") or "")
        if not i.get("in_latest_scan"):
            parts = ["not in the latest scan"]
        elif not i.get("ranked"):
            parts = ["scanned, not a ranked HSF setup"]
            if i.get("previous_rank") is not None:
                parts.append(f"was ranked #{i['previous_rank']} on the previous scan")
        else:
            parts = [f"HSF Score {_fmt(i.get('hsf_score'))}"]
            if i.get("score_change") is not None:
                parts.append(f"change {float(i['score_change']):+.0f} since previous scan")
            elif i.get("previous_hsf_score") is None:
                parts.append("newly ranked this scan")
            if i.get("rank") is not None:
                parts.append(f"rank #{i['rank']}")
            if i.get("setup"):
                parts.append(f"setup {i['setup']}")
            if i.get("status"):
                parts.append(f"status {i['status']}")
            if i.get("fading"):
                parts.append("fading")
        if i.get("price_change_pct") is not None and _fmt(i.get("price_change_pct"), 1) is not None:
            parts.append(f"price change {float(i['price_change_pct']):+.1f}%")
        if i.get("rvol") is not None and _fmt(i.get("rvol"), 1) is not None:
            parts.append(f"relative volume {float(i['rvol']):.1f}x")
        e = earnings.get(t.upper())
        if e:
            days = int(e.get("earnings_days") or 0)
            parts.append("reports earnings " + ("today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"))
        lines.append(f"- {t}: " + "; ".join(parts))
    extra = len(items) - WATCHLIST_NOTE_MAX_SYMBOLS
    if extra > 0:
        lines.append(f"- ({extra} more symbols not listed)")
    return "\n".join(lines)


def watchlist_note(username: str, watchlist: Dict[str, Any], entitlements: Dict[str, bool]) -> Dict[str, Any]:
    """Claude's plain-English rundown of what changed on one of your watchlists since the
    previous market scan. Cached per watchlist and scan, so asking again costs nothing."""
    import datetime as dt

    from api import alert_rules, watchlist_intel

    intel = watchlist_intel.intelligence(username, watchlist, entitlements)
    tickers = [str(i.get("ticker") or "") for i in intel.get("items") or []]
    earnings = alert_rules.earnings_lookup(tickers, dt.datetime.now(dt.timezone.utc)) if tickers else {}
    earnings = {t: e for t, e in earnings.items()
                if e.get("earnings_days") is not None and int(e["earnings_days"]) <= 7}
    facts = watchlist_facts(intel, earnings)
    as_of = intel.get("last_scan_at")
    if not facts:
        return {"snapshot_time": as_of, "text": None}
    name = str(watchlist.get("name") or "this watchlist")

    def make() -> str:
        return _ask(system=WATCHLIST_NOTE_SYSTEM,
                    user=(f"Watchlist \"{name}\". Latest scan {as_of}; previous scan {intel.get('previous_scan_at')}.\n"
                          f"Facts per symbol:\n{facts}\n\nWrite the rundown."),
                    max_tokens=350, username=username, feature="watchlist_note")

    key = ("watchlist", username, int(watchlist["id"]), str(as_of), hash(facts))
    return {"snapshot_time": as_of, "text": _shared(key, make)}
