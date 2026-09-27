"""Run 85 — one responsibility boundary for every customer-facing Claude answer.

HSF's deterministic scoring system produces the HSF Score and the opportunity
ranking; Claude only explains data it is given. These rules are appended to the
system prompt of every analysis feature (via ui.ai.ask_claude/ask_claude_chat),
and a narrow output check removes any line that still gives a trade instruction.
The prompt is the primary fix; the output check is defense in depth.

JSON-returning features (natural-language screener / scan setup) and the admin
error triage are exempt: they don't produce customer-facing market commentary.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

EXEMPT_FEATURES = frozenset({"nl_screener", "nl_scan_setup", "admin_triage"})

RULES = (
    "\n\nBoundaries (always apply): HSF's own scoring system calculated the HSF Score "
    "and the ranking/order you are given; never say you calculated, ranked, picked or "
    "chose them, and keep the order provided. Describe the observed data, separate facts "
    "from your interpretation, and mention conflicting evidence and risks. Do not invent "
    "data or infer things not in it (e.g. institutional buying). Never tell the reader to "
    "buy, sell, hold, short, enter, exit, size a position or take/avoid a trade, and never "
    "give entry/exit levels, stops or price targets. Do not predict or promise outcomes. "
    "This is general research commentary, not personalized investment advice."
)

# Lines that still read as a trade instruction or a forward-looking promise.
_ACTION = re.compile(
    r"(?<![-\w])(buy|sell)(?![-\w])|\bshort (it|this|them)\b|\bgo short\b"
    r"|\b(enter|exit)\b|\bentry\b|\bentries\b"
    r"|\bposition[- ]siz|\bsize (a|the|your) position"
    r"|\b(take|avoid) (the|a|this) (trade|position)"
    r"|\bstop[- ]loss|\bprice target|\btarget price"
    r"|\bwill (rise|rally|climb|break ?out|outperform|go up|move higher)\b"
    r"|\bguarantee"
    r"|\binstitutional (buying|accumulation|participation|interest)",
    re.IGNORECASE,
)
REMOVED_NOTE = "_Some wording was removed: HSF explains setups but doesn't give trade instructions._"


def applies(feature: Optional[str]) -> bool:
    return str(feature or "") not in EXEMPT_FEATURES


def with_rules(system: str) -> str:
    return (system or "") + RULES


def find_action_language(text: str) -> list[str]:
    """Lines of `text` that contain trade-instruction / promissory wording."""
    return [line for line in str(text or "").splitlines() if _ACTION.search(line)]


def scrub(text: Optional[str]) -> Tuple[Optional[str], int]:
    """Drop lines that give trade instructions; returns (text, lines_removed).
    Market facts on other lines are left untouched."""
    if not text:
        return text, 0
    kept, removed = [], 0
    for line in str(text).splitlines():
        if _ACTION.search(line):
            removed += 1
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    if removed:
        out = (out + "\n\n" + REMOVED_NOTE).strip()
    return (out or None), removed
