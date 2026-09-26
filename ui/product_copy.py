"""Run 62 — canonical HSF product copy (positioning, disclaimers, claim guard).

One source for how HSF describes itself, so the landing page, methodology page,
emails and page titles say the same factual thing. No performance language:
HSF's forward research has not validated any effectiveness claim yet, so copy
describes what the product does, never how well it does it.

Pure Python — safe to import without Streamlit.
"""
from __future__ import annotations

import re
from typing import List

PRODUCT_NAME = "HSF AI Stock Scanner"
BRAND = "HSFinest.AI"

# The headline promise, used as the hero line and page subtitle.
TAGLINE = "Know what matters in the market right now."

# One line: what HSF is.
POSITIONING_ONE_LINE = (
    "HSF continuously scans the U.S. stock market and helps traders identify, "
    "understand and monitor noteworthy technical setups."
)

# Short description: the landing hero body.
POSITIONING_SHORT = (
    "HSF continuously scans thousands of tradable U.S. stocks and organizes "
    "noteworthy setups so you can focus your research."
)

# Long description: methodology intro, about text.
POSITIONING_LONG = (
    "HSF scans a broad universe of tradable U.S.-listed stocks throughout the "
    "trading day. It ranks the setups that stand out, shows the technical "
    "evidence behind each one, and tracks how they change through the session. "
    "HSF is a research tool: it helps you decide what deserves a closer look, "
    "and leaves the decision to you."
)

# Browser-tab title for the main app.
PAGE_TITLE = f"{PRODUCT_NAME} · {BRAND}"

# Sign-off line used in emails in place of the old promotional tagline.
EMAIL_SIGNOFF = f"{BRAND} — {TAGLINE}"

# The four things HSF does (landing "What HSF does" row).
PILLARS = (
    ("Scan", "Continuously monitors thousands of tradable U.S. stocks through the trading day."),
    ("Rank", "Surfaces the setups that stand out right now."),
    ("Explain", "Shows the technical evidence behind each setup."),
    ("Track", "Shows how setups change through the session."),
)

# Trust points (landing "Why you can trust what you see").
TRUST_POINTS = (
    ("Fresh market data", "Every result shows when it was scanned."),
    ("Transparent explanations", "Each setup lists the evidence that put it there."),
    ("Point-in-time scanning", "Results are recorded as the scanner saw them, never revised with hindsight."),
    ("Research-first methodology", "HSF measures its own methods forward before making claims about them."),
)

DISCLAIMER = (
    "HSF is a market-research and decision-support tool, not financial advice. "
    "It does not recommend buying or selling any security."
)

# Standard label and sentence for historical/experimental analytics.
HISTORICAL_RESEARCH_LABEL = "Historical research"
HISTORICAL_RESEARCH_NOTE = (
    "Historical observations are descriptive and do not represent validated "
    "forward performance."
)

# Promotional language HSF must not use until forward evidence supports it.
# Matched case-insensitively on word boundaries in user-facing copy.
PROHIBITED_CLAIM_PATTERNS = (
    r"\bwin\b",
    r"\bwinning\b",
    r"\bwinners?\b",
    r"\bbeat(s)? the market\b",
    r"\bproven\b",
    r"\bproven edge\b",
    r"\bhigh win rate\b",
    r"\bguarantee(d|s)?\b",
    r"\bbest stocks\b",
    r"\bbest-performing\b",
    r"\bprofitable signals?\b",
    r"\bpays? off\b",
)
_PROHIBITED_RE = re.compile("|".join(PROHIBITED_CLAIM_PATTERNS), re.IGNORECASE)

# Phrases that negate a claim ("does not guarantee") are allowed: disclaimers
# legitimately contain the words the guard looks for.
_NEGATION_RE = re.compile(r"\b(not|no|never|doesn't|does not|don't|do not|without)\b[^.]{0,40}$",
                          re.IGNORECASE)


def find_prohibited_claims(text: str) -> List[str]:
    """Return the promotional-claim phrases found in `text` (empty = clean).

    A match preceded closely in the same sentence by a negation ("past
    performance does not guarantee…") is a disclaimer, not a claim, and is
    ignored.
    """
    hits: List[str] = []
    for m in _PROHIBITED_RE.finditer(text or ""):
        before = (text or "")[max(0, m.start() - 60):m.start()]
        if _NEGATION_RE.search(before):
            continue
        hits.append(m.group(0))
    return hits
