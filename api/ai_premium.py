"""Premium AI tools (2026-10-23, can_ai_notes).

- Ask HSF: one chat that answers from the account's watchlists, the latest market scan,
  earnings, news headlines and HSF's matured track record.
- Journal coach: a short note on patterns in the account's own closed trades.
- Scan builder: plain English -> Custom scan settings (the saved-view shape).
- "Why it fired": one AI line added to Premium alert-rule events.
- Morning email rundown: the watchlist rundown (api.ai.watchlist_note) in the digest.

Every call goes through ui.ai.ask_claude / ask_claude_chat like the other AI features
(kill switch AI_ENABLED, the per-account daily cap, timeout, the Run 85 rules).
Claude only gets facts HSF already has; prompts tell it not to invent anything.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from api.ai import MAX_CHAT_TURNS, MAX_TURN_CHARS, _ask, _fmt, _raise_for

log = logging.getLogger("hsf.api")

# ---- shared fact lines ------------------------------------------------------------------------------
_TICKER_RE = re.compile(r"\$?\b([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")
_NOT_TICKERS = frozenset({
    "A", "I", "AM", "AN", "AND", "ANY", "ARE", "AS", "AT", "BE", "BUT", "BY", "CAN", "DO", "FOR", "HOW", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "NOT", "OF", "ON", "OR", "SO", "THE", "TO", "UP", "US", "WE", "WHAT",
    "WHO", "WHY", "YOU", "HSF", "RVOL", "EMA", "SPY", "QQQ", "ETF", "IPO", "CEO", "EPS", "AI", "OK", "VS",
})
MAX_MENTIONS = 5
TOP_SCAN_ROWS = 10
EARNINGS_DAYS = 7


def _earn_text(e: Optional[Dict[str, Any]]) -> Optional[str]:
    if not e or e.get("earnings_days") is None:
        return None
    d = int(e["earnings_days"])
    when = "today" if d == 0 else "tomorrow" if d == 1 else f"in {d} days"
    t = str(e.get("earnings_time") or "").lower()
    return f"reports earnings {when}" + (" before the open" if t == "bmo" else " after the close" if t == "amc" else "")


def view_line(v: Dict[str, Any], earnings: Optional[Dict[str, Any]] = None) -> str:
    """One line of scan facts for a ticker_view (api.watchlist_intel.ticker_view)."""
    t = str(v.get("ticker") or "")
    if not v.get("present"):
        parts = ["not in the latest scan"]
    elif not v.get("ranked"):
        parts = ["scanned, not a ranked HSF setup"]
    else:
        parts = [f"HSF Score {_fmt(v.get('hsf_score'))}", f"rank #{v.get('rank')}"]
        if v.get("previous_hsf_score") is not None and v.get("hsf_score") is not None:
            parts.append(f"change {float(v['hsf_score']) - float(v['previous_hsf_score']):+.0f} since previous scan")
        if v.get("setup"):
            parts.append(f"setup {v['setup']}")
        if v.get("status"):
            parts.append(f"status {v['status']}")
        if v.get("fading"):
            parts.append("fading")
        if v.get("prebreakout"):
            parts.append("PreBreakout candidate")
    if _fmt(v.get("price"), 2) is not None:
        parts.append(f"last ${float(v['price']):.2f}")
    if _fmt(v.get("price_change_pct"), 1) is not None:
        parts.append(f"price change {float(v['price_change_pct']):+.1f}%")
    if _fmt(v.get("rvol"), 1) is not None:
        parts.append(f"relative volume {float(v['rvol']):.1f}x")
    e = _earn_text(earnings)
    if e:
        parts.append(e)
    return f"- {t}: " + "; ".join(parts)


def mentioned_tickers(text: str, known: Iterable[str]) -> List[str]:
    """Tickers named in a question: a $TSLA anywhere, or an upper-case word that HSF knows
    (in the scan or the user's lists). Plain words like 'AND' are never tickers."""
    known_set = {str(k).upper() for k in known}
    out: List[str] = []
    for m in _TICKER_RE.finditer(text or ""):
        raw = m.group(0)
        sym = m.group(1).upper()
        if sym in _NOT_TICKERS or sym in out:
            continue
        if raw.startswith("$") or (m.group(1).isupper() and sym in known_set):
            out.append(sym)
        if len(out) >= MAX_MENTIONS:
            break
    return out


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # a missing source drops that block, never the answer
        log.info(json.dumps({"event": "ai_context_part_failed", "error": type(e).__name__}))
        return default


# ---- Ask HSF ----------------------------------------------------------------------------------------
ASK_SYSTEM = (
    "You are Ask HSF, the research assistant inside the HSF stock scanner, talking with one trader. "
    "Answer from the HSF facts provided (their watchlists, the latest market scan, earnings dates, recent "
    "news headlines, HSF's matured track record and the market brief). If the facts don't cover the question, "
    "say so plainly and name the HSF page that would show it (Scanner, Stocks, Watchlists, Alerts, Track record, "
    "Brief). Never invent prices, news, levels or tickers. Be concise: under 150 words unless a short list is "
    "clearer. Plain text; short bullets are fine; no headings or emojis."
)


def _watchlists_block(username: str, entitlements: Dict[str, bool], obs, earnings_for) -> tuple:
    """(text, tickers): every list's name and symbols, and fact lines for the default list."""
    from api import user_data, watchlist_intel

    lists = user_data.list_watchlists(username)
    if not lists:
        return "The trader has no watchlists yet.", []
    default = next((w for w in lists if w.get("is_default")), lists[0])
    all_tickers: List[str] = []
    lines = []
    for w in lists[:10]:
        wl = user_data.get_watchlist(username, int(w["id"]))
        syms = [str(i["ticker"]).upper() for i in wl.get("items") or []]
        all_tickers += syms
        tag = " (default)" if w["id"] == default["id"] else ""
        lines.append(f"- \"{w['name']}\"{tag}: {', '.join(syms[:40]) or 'empty'}")
    default_syms = [str(i["ticker"]).upper() for i in
                    user_data.get_watchlist(username, int(default["id"])).get("items") or []][:40]
    earn = earnings_for(all_tickers)
    premium = bool(entitlements.get("can_early_breakout"))
    facts = [view_line(watchlist_intel.ticker_view(obs, t, premium=premium), earn.get(t)) for t in default_syms] if obs else []
    others = sorted({t for t in all_tickers if t in earn and t not in default_syms})
    extra = [f"- {t}: {_earn_text(earn[t])}" for t in others]
    text = "Watchlists:\n" + "\n".join(lines)
    if facts:
        text += f"\n\nDefault list \"{default['name']}\", latest scan facts:\n" + "\n".join(facts)
    if extra:
        text += "\n\nEarnings this week on other lists:\n" + "\n".join(extra)
    return text, sorted(set(all_tickers))


def _top_block(obs, entitlements: Dict[str, bool], earn: Dict[str, Dict[str, Any]]) -> str:
    from api import watchlist_intel

    premium = bool(entitlements.get("can_early_breakout"))
    ranked = sorted(obs["ranked"].items(), key=lambda kv: int(kv[1].get("rank") or 10**6))[:TOP_SCAN_ROWS]
    rows = [view_line(watchlist_intel.ticker_view(obs, t, premium=premium), earn.get(t)) for t, _ in ranked]
    return f"Top {len(rows)} ranked setups in the latest market scan (HSF order):\n" + "\n".join(rows)


def _ticker_block(t: str, obs, entitlements: Dict[str, bool], earn: Dict[str, Dict[str, Any]],
                  with_evidence: bool) -> str:
    from api import news, watchlist_intel
    from api.outcomes import ai_evidence

    premium = bool(entitlements.get("can_early_breakout"))
    parts = [view_line(watchlist_intel.ticker_view(obs, t, premium=premium), earn.get(t)) if obs else f"- {t}"]
    items = (_safe(lambda: news.stock_news(t, limit=3), {}) or {}).get("items") or []
    for n in items[:3]:
        parts.append(f"  news {str(n.get('published_at'))[:10]}: {n.get('headline')}")
    if with_evidence:
        ev = _safe(lambda: ai_evidence(t))
        if ev:
            parts.append("  " + ev.replace("\n", "\n  "))
    return "\n".join(parts)


def ask_context(username: str, entitlements: Dict[str, bool], question: str,
                now: Optional[dt.datetime] = None) -> str:
    from api import watchlist_intel
    from api.alert_rules import earnings_lookup

    now = now or dt.datetime.now(dt.timezone.utc)

    def earnings_for(tickers: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        got = _safe(lambda: earnings_lookup(tickers, now), {}) or {}
        return {t: e for t, e in got.items() if e.get("earnings_days") is not None
                and int(e["earnings_days"]) <= EARNINGS_DAYS}

    obs = _safe(watchlist_intel.market_observation)
    blocks = [f"Now: {now.astimezone(dt.timezone.utc):%Y-%m-%d %H:%M} UTC."]
    if obs:
        blocks.append(f"Latest market scan: {obs.get('scan_at')}.")
    else:
        blocks.append("No market scan is available right now.")
    wl_text, wl_tickers = _safe(lambda: _watchlists_block(username, entitlements, obs, earnings_for),
                                ("Watchlists are unavailable right now.", []))
    blocks.append(wl_text)
    known = set(wl_tickers) | set((obs or {}).get("raw", {}).keys()) | set((obs or {}).get("ranked", {}).keys())
    if obs:
        top_syms = [t for t, _ in sorted(obs["ranked"].items(), key=lambda kv: int(kv[1].get("rank") or 10**6))[:TOP_SCAN_ROWS]]
        blocks.append(_safe(lambda: _top_block(obs, entitlements, earnings_for(top_syms)), ""))
    mentions = mentioned_tickers(question, known)
    if mentions:
        earn = earnings_for(mentions)
        details = [_safe(lambda t=t, i=i: _ticker_block(t, obs, entitlements, earn, with_evidence=i < 2), f"- {t}")
                   for i, t in enumerate(mentions)]
        blocks.append("Stocks named in the question:\n" + "\n".join(details))
    brief = _safe(_brief_facts)
    if brief:
        blocks.append("Market brief facts:\n" + brief)
    return "\n\n".join(b for b in blocks if b)


def _brief_facts() -> str:
    from api.market import _brief_core
    from ui.market_brief import _brief_narrative_facts

    data = (_brief_core() or {}).get("data")
    return _brief_narrative_facts(data) if data else ""


def ask(username: str, entitlements: Dict[str, bool], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """POST /v1/ai/ask. The client keeps the history; the server caps it and puts fresh
    facts (for the newest question) first."""
    from ui.ai import ask_claude_chat

    turns = [{"role": m["role"], "content": str(m["content"])[:MAX_TURN_CHARS]} for m in messages][-MAX_CHAT_TURNS * 2:]
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    question = turns[-1]["content"] if turns else ""
    context = ask_context(username, entitlements, question)
    api_messages = [{"role": "user", "content": f"HSF facts for this conversation:\n\n{context}"},
                    {"role": "assistant", "content": "Got it. I'll answer from these HSF facts only."}, *turns]
    answer, err = ask_claude_chat(system=ASK_SYSTEM, messages=api_messages, max_tokens=600,
                                  username=username, feature="ask_hsf")
    if not answer:
        _raise_for(err)
    return {"answer": answer}


# ---- journal coach ----------------------------------------------------------------------------------
COACH_MIN_TRADES = 5
COACH_MAX_TRADES = 40
COACH_SYSTEM = (
    "You coach one trader on their own trade journal. From the closed-trade facts given, write 3-5 short "
    "bullet points (each starting with '- ') about patterns in THEIR past results: how results differ by HSF "
    "Score at the time they took the trade, by holding time and by where the trade came from; whether average "
    "wins are larger or smaller than average losses; streaks or one trade that dominates the total. End with one "
    "bullet naming a habit worth reviewing in their next trades. Use only the numbers given and quote them. "
    "Small samples (under 5 trades in a group) must be called small. Do not recommend specific stocks, give "
    "price levels or predict markets. Say 'when you took the trade' and 'when you closed it' rather than entry, "
    "exit, buy or sell. Plain, encouraging, direct. No headings, no emojis."
)


def journal_facts(ins: Dict[str, Any], trades: List[Dict[str, Any]]) -> str:
    s = ins.get("summary") or {}
    lines = [f"Closed trades: {ins.get('closed')}. Wins {s.get('wins')}, losses {s.get('losses')}, "
             f"win rate {s.get('win_rate')}%. Average return {s.get('avg_return_pct')}% "
             f"(average win {s.get('avg_win_pct')}%, average loss {s.get('avg_loss_pct')}%). "
             f"Profit factor {s.get('profit_factor')}. Total P&L ${s.get('total_pnl')}. "
             f"Average hold {s.get('avg_hold_days')} days."]
    for title, key in (("By HSF Score when taken", "by_score"), ("By holding time", "by_hold"),
                       ("By where the trade came from", "by_source")):
        groups = ins.get(key) or []
        if groups:
            lines.append(title + ":")
            lines += [f"- {g['label']}: {g['count']} trades, win rate {g['win_rate']}%, "
                      f"average {g['avg_return_pct']}%, P&L ${g['pnl']}" for g in groups]
    if trades:
        lines.append(f"Most recent {len(trades)} trades (oldest first):")
        lines += [f"- {t['ticker']}: {t['return_pct']:+.1f}%, held {t['hold']}, {t['source']}, "
                  f"HSF Score when taken {t['score']}" for t in trades]
    return "\n".join(lines)


def journal_coach(username: str) -> Dict[str, Any]:
    """POST /v1/ai/journal-coach. text is null below COACH_MIN_TRADES closed trades.
    Cached per account and journal state, so asking again costs nothing."""
    from api import trading
    from api.ai import _shared
    from db.trades import closed_trades

    ins = trading.insights(username)
    closed = int(ins.get("closed") or 0)
    if closed < COACH_MIN_TRADES:
        return {"closed": closed, "min_trades": COACH_MIN_TRADES, "text": None}
    recent = []
    for t in closed_trades(username)[-COACH_MAX_TRADES:]:
        entry, exit_ = float(t.get("entry_price") or 0), t.get("exit_price")
        if not entry or exit_ is None:
            continue
        o, c = trading._ts(t.get("entered_at")), trading._ts(t.get("closed_at"))
        days = (c - o).total_seconds() / 86400.0 if o and c else None
        recent.append({"ticker": t.get("ticker"), "return_pct": (float(exit_) - entry) / entry * 100.0,
                       "hold": trading._hold_band(days).lower(),
                       "source": trading.SOURCE_LABELS.get(str(t.get("source") or "api"), str(t.get("source"))).lower(),
                       "score": _fmt(t.get("hsf_score")) or "not recorded"})
    facts = journal_facts(ins, recent)

    def make() -> str:
        return _ask(system=COACH_SYSTEM, user=f"Journal facts:\n{facts}\n\nWrite the coaching note.",
                    max_tokens=450, username=username, feature="journal_coach")

    text = _shared(("coach", username, hash(facts)), make)
    return {"closed": closed, "min_trades": COACH_MIN_TRADES, "text": text}


# ---- scan builder -----------------------------------------------------------------------------------
BUILDER_UNIVERSES = ("sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker", "earnings")
BUILDER_SESSIONS = ("regular", "premarket", "afterhours")
BUILDER_WINDOWS = ("today", "tomorrow", "week")
BUILDER_TOP_N = (10, 25, 50, 100, 200)
BUILDER_SYSTEM = (
    "You turn a trader's plain-English scan request into settings for HSF's Custom scan. Reply with ONE JSON "
    "object and nothing else, with these keys (leave a key out when the request doesn't set it):\n"
    "universe: one of sp500, nasdaq, combo (S&P 500 + NASDAQ), us_market (whole U.S. market), watchlist, ticker, "
    "earnings (stocks reporting earnings)\n"
    "ticker: a symbol when universe is ticker\n"
    "watchlist_id: an id from the trader's watchlists when universe is watchlist\n"
    "earnings_window: today, tomorrow or week, when universe is earnings\n"
    "session: regular, premarket or afterhours\n"
    "top_n: 10, 25, 50, 100 or 200 results\n"
    "min_price, max_price: dollars\n"
    "min_dollar_vol: minimum average daily dollar volume in dollars (e.g. 5000000 for $5M)\n"
    "unusual: true to keep only unusual relative volume\n"
    "gap: true to keep only gappers; min_gap: minimum gap in percent\n"
    "score_all: true to score every ticker, not only setups (ticker, watchlist or earnings universes)\n"
    "explain: one short sentence saying what the scan will look for, and naming any part of the request "
    "these settings can't express (describe filters only; never words like buy, sell, entry or exit).\n"
    "Put the whole JSON object on one line.\n"
    "Default universe is sp500 when none is implied. Use only these keys."
)


def _num_or_none(v: Any, lo: float, hi: float) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and lo <= f <= hi else None


def parse_builder(text: str, watchlist_ids: Sequence[int]) -> Dict[str, Any]:
    """The model's JSON -> a validated Custom scan view (the saved-view shape the web
    applies). Unknown keys and bad values are dropped, never passed through."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object")
    raw = json.loads(m.group(0))
    if not isinstance(raw, dict):
        raise ValueError("not an object")
    view: Dict[str, Any] = {"name": "AI scan", "ticker": ""}
    u = str(raw.get("universe") or "sp500").strip().lower()
    view["universe"] = u if u in BUILDER_UNIVERSES else "sp500"
    if view["universe"] == "ticker":
        t = str(raw.get("ticker") or "").strip().upper().lstrip("$")
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,9}", t):
            view["ticker"] = t
        else:
            view["universe"] = "sp500"
    if view["universe"] == "watchlist":
        try:
            wid = int(raw.get("watchlist_id"))
        except (TypeError, ValueError):
            wid = None
        if wid in set(int(i) for i in watchlist_ids):
            view["watchlist_id"] = wid
        elif watchlist_ids:
            view["watchlist_id"] = int(watchlist_ids[0])
        else:
            view["universe"] = "sp500"
    w = str(raw.get("earnings_window") or "today").strip().lower()
    view["earnings_window"] = w if w in BUILDER_WINDOWS else "today"
    s = str(raw.get("session") or "regular").strip().lower()
    view["session"] = s if s in BUILDER_SESSIONS else "regular"
    try:
        n = int(raw.get("top_n") or 25)
    except (TypeError, ValueError):
        n = 25
    view["top_n"] = min(BUILDER_TOP_N, key=lambda x: abs(x - n))
    for k, hi in (("min_price", 1e6), ("max_price", 1e6), ("min_dollar_vol", 1e12), ("min_gap", 100.0)):
        v = _num_or_none(raw.get(k), 0.0, hi)
        if v is not None:
            view[k] = v
    if "min_price" in view and "max_price" in view and view["min_price"] > view["max_price"]:
        view["min_price"], view["max_price"] = view["max_price"], view["min_price"]
    view["unusual"] = raw.get("unusual") is True
    view["gap"] = raw.get("gap") is True or "min_gap" in view
    view["score_all"] = raw.get("score_all") is True and view["universe"] in ("ticker", "watchlist", "earnings")
    explain = str(raw.get("explain") or "").strip()[:300]
    return {"view": view, "explanation": explain or None}


def scan_builder(username: str, text: str) -> Dict[str, Any]:
    """POST /v1/ai/scan-builder: settings to fill the Custom scan form (nothing runs)."""
    from api import user_data

    lists = _safe(lambda: user_data.list_watchlists(username), []) or []
    wl = "\n".join(f"- id {w['id']}: \"{w['name']}\"" + (" (default)" if w.get("is_default") else "") for w in lists)
    user = (f"Trader's watchlists:\n{wl or '- none'}\n\nRequest: {text.strip()[:500]}")
    reply = _ask(system=BUILDER_SYSTEM, user=user, max_tokens=300, username=username, feature="scan_builder")
    try:
        return parse_builder(reply, [int(w["id"]) for w in lists])
    except (ValueError, TypeError):
        from api.ai import AIFailed

        raise AIFailed("AI couldn't turn that into scan settings. Try rewording it.")


# ---- "why it fired" ---------------------------------------------------------------------------------
WHY_SYSTEM = (
    "You add one short sentence (under 30 words) to a stock alert explaining what in the scan facts stands out "
    "about this move: the setup, score or rank change, relative volume or an earnings date. Use only the facts "
    "given. No advice, no predictions, no preamble, no emojis."
)
WHY_PER_PASS = 20


def why_enabled() -> bool:
    return os.getenv("HSF_ALERT_AI_WHY", "1").strip() != "0"


def why_daily_cap() -> int:
    try:
        return max(0, int(os.getenv("HSF_ALERT_AI_WHY_DAILY", "10")))
    except ValueError:
        return 10


def _why_used_today(user: str) -> int:
    try:
        from db.ai_usage import ai_calls_today

        return ai_calls_today(user, feature="alert_why")
    except Exception:
        return 0


def alert_why(event: Dict[str, Any], view: Dict[str, Any]) -> Optional[str]:
    """One sentence for one event, or None. Doesn't use the account's daily AI budget
    (it has its own cap, HSF_ALERT_AI_WHY_DAILY); never raises."""
    from ui.ai import ask_claude

    facts = f"Alert: {event.get('message')}\nScan facts:\n{view_line(view, view if view.get('earnings_days') is not None else None)}"
    text, _err = ask_claude(system=WHY_SYSTEM, user=facts, max_tokens=90, username=None, feature="alert_why")
    if not text:
        return None
    line = " ".join(text.split())
    if line.startswith("_Some wording was removed"):
        return None
    try:
        from db.ai_usage import record_ai_call

        record_ai_call(str(event.get("user_id") or ""), "alert_why")
    except Exception:
        pass
    return line[:280]


def annotate_alerts(events: List[Dict[str, Any]], obs: Dict[str, Any]) -> int:
    """Add 'Why: ...' to Premium events' messages before they are stored and delivered.
    Capped per pass and per account per day; any failure leaves the message as it was."""
    if not events or not why_enabled():
        return 0
    from api import watchlist_intel
    from ui.ai import is_configured

    if not is_configured():
        return 0

    cap = why_daily_cap()
    used: Dict[str, int] = {}
    done = 0
    for ev in events:
        if done >= WHY_PER_PASS:
            break
        if str(ev.get("_tier") or "") not in ("premium", "admin"):
            continue
        user = str(ev.get("user_id") or "")
        if user not in used:
            used[user] = _why_used_today(user)
        if used[user] >= cap:
            continue
        try:
            view = ev.get("_view") or watchlist_intel.ticker_view(obs, str(ev["ticker"]), premium=True)
            line = alert_why(ev, view)
        except Exception:
            line = None
        if line:
            ev["message"] = f"{ev['message']}\nWhy: {line}"
            used[user] += 1
            done += 1
    return done


# ---- morning email rundown --------------------------------------------------------------------------
def digest_enabled() -> bool:
    return os.getenv("HSF_DIGEST_AI_RUNDOWN", "1").strip() != "0"


def digest_rundown(email: str) -> Optional[str]:
    """The default watchlist's AI rundown for the morning email, or None (AI off, no
    list, no scan, limit reached). Never raises: the digest goes out without it."""
    if not digest_enabled():
        return None
    try:
        from api import ai, user_data
        from domain.plans import compute_entitlements, has_min_tier

        lists = user_data.list_watchlists(email)
        if not lists:
            return None
        default = next((w for w in lists if w.get("is_default")), lists[0])
        wl = user_data.get_watchlist(email, int(default["id"]))
        if not wl.get("items"):
            return None
        ent = compute_entitlements(tier_obj="premium", is_admin=False, has_min_tier_fn=has_min_tier)
        return (ai.watchlist_note(email, wl, ent) or {}).get("text")
    except Exception as e:
        log.info(json.dumps({"event": "digest_ai_rundown_skipped", "error": type(e).__name__}))
        return None
