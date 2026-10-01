"""Run 62 — signed-out landing experience.

Replaces the old "giant logo + login form" first screen with:

  hero (small logo, product name, tagline, one-paragraph description)
  → sign-in / create-account form (unchanged auth logic, in ui.auth)
  → What HSF does · example result layout · why you can trust it · disclaimer

The hero is sized to leave the sign-in form inside the first phone viewport.
All copy comes from ui.product_copy (factual, no performance claims). The
example results are clearly labelled illustrations with placeholder names —
never live data, never Run 56 research data.
"""
from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

import ui.product_copy as _product_copy
from ui.acquisition import track_landing_visit_once
from ui.pricing import plans_html
from ui.product_copy import (
    DISCLAIMER,
    PILLARS,
    POSITIONING_SHORT,
    PRODUCT_NAME,
    TAGLINE,
    TRUST_POINTS,
)

# getattr: a stale ui.product_copy on Streamlit Cloud must not break the landing page.
HSF_SCORE_ONE_LINE = getattr(
    _product_copy, "HSF_SCORE_ONE_LINE",
    "HSF Score (0–100) ranks setups by how strongly the current technical evidence lines up. "
    "It is not a probability of profit or a prediction.",
)

ROOT = Path(__file__).resolve().parents[1]

# Illustrative only: placeholder names and example evidence lines in the same
# format the scanner's "Why" column uses. Not live data, not recommendations.
EXAMPLE_ROWS: List[Tuple[str, int, str, Tuple[str, ...]]] = [
    ("Stock A", 82, "Momentum breakout", ("2.4× avg volume", "at 20d high", "outpacing SPY")),
    ("Stock B", 74, "Gap and go", ("+4.1% gap", "1.9× avg volume")),
    ("Stock C", 67, "Trend continuation", ("+9% over 10d", "earnings in 3d ⚠️")),
]
# Streamlit serves pages/methodology.py at /methodology.
METHODOLOGY_HREF = "/methodology"
EXAMPLE_NOTE = "Example of how results look. Illustrative names and values, not live data or recommendations."

_CSS = """
<style>
/* .hsf-hero top margin clears Streamlit's 60px header bar, which clipped the title. */
.hsf-land{--hsf-gold:#b8892b;--hsf-line:rgba(128,128,128,.25);--hsf-soft:rgba(128,128,128,.08)}
.hsf-hero{display:flex;align-items:center;gap:18px;margin:2.75rem 0 6px}
.hsf-hero img{width:104px;height:auto;flex:0 0 auto}
.hsf-hero h1{font-size:clamp(1.55rem,4.2vw,2.3rem);line-height:1.1;margin:0 0 4px;padding:0}
.hsf-hero .hsf-tag{font-size:clamp(1.02rem,2.6vw,1.25rem);font-weight:600;margin:0 0 6px;color:var(--hsf-gold)}
.hsf-hero p{margin:0;max-width:62ch;opacity:.9}
.hsf-cta-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:12px 0 0}
.hsf-cta{display:inline-block;border-radius:8px;background:var(--hsf-gold);color:#111!important;
  padding:8px 12px;text-decoration:none;font-weight:700}
.hsf-cta-hint{font-size:.9rem;opacity:.8;margin:0}
.hsf-sec h2{font-size:1.2rem;margin:22px 0 10px;padding:0}
.hsf-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.hsf-card{border:1px solid var(--hsf-line);border-radius:10px;padding:12px 14px;background:var(--hsf-soft)}
.hsf-card h3{font-size:1rem;margin:0 0 4px;padding:0}
.hsf-card p{margin:0;font-size:.9rem;opacity:.85;line-height:1.4}
.hsf-ex{display:flex;flex-direction:column;gap:8px}
.hsf-row{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;border:1px solid var(--hsf-line);
  border-radius:10px;padding:10px 12px}
.hsf-row .t{font-weight:700;min-width:5.5rem}
.hsf-row .s{font-variant-numeric:tabular-nums;font-weight:600}
.hsf-row .k{opacity:.8;font-size:.9rem}
.hsf-chip{font-size:.8rem;border:1px solid var(--hsf-line);border-radius:999px;padding:2px 8px;white-space:nowrap}
.hsf-note{font-size:.8rem;opacity:.7;margin:6px 0 0}
.hsf-score{font-size:.95rem;margin:10px 0 0;max-width:62ch}
.hsf-disc{font-size:.85rem;opacity:.8;margin:18px 0 4px}
@media (max-width:640px){.hsf-hero{margin-top:4px}}  /* phones: the page padding already clears the header */
@media (max-width:520px){
  .hsf-hero{gap:12px;align-items:flex-start}
  .hsf-hero img{width:64px;margin-top:4px}
}
</style>
"""


@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    try:
        from ui.header import _LOGO_CANDIDATES

        # Resolve against the repo root so the logo loads whatever the cwd is.
        path = next(ROOT / c for c in _LOGO_CANDIDATES if (ROOT / c).exists())
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""


def hero_html(logo_uri: str = "") -> str:
    img = (f'<img src="{logo_uri}" alt="HSFinest.AI logo">' if logo_uri else "")
    return (
        f'{_CSS}<div class="hsf-land"><div class="hsf-hero">{img}<div>'
        f'<h1>{html.escape(PRODUCT_NAME)}</h1>'
        f'<p class="hsf-tag">{html.escape(TAGLINE)}</p>'
        f'<p>{html.escape(POSITIONING_SHORT)}</p>'
        '<div class="hsf-cta-row"><a class="hsf-cta" href="#hsf-signup">Start scanning free</a>'
        '<p class="hsf-cta-hint">Sign in or create a free account. No credit card required.</p></div>'
        '</div></div></div>'
    )


def details_html() -> str:
    pillars = "".join(
        f'<div class="hsf-card"><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p></div>' for t, d in PILLARS
    )
    rows = "".join(
        '<div class="hsf-row">'
        f'<span class="t">{html.escape(name)}</span>'
        f'<span class="s" aria-label="HSF Score {score}">HSF Score {score}</span>'
        f'<span class="k">{html.escape(setup)}</span>'
        + "".join(f'<span class="hsf-chip">{html.escape(w)}</span>' for w in why)
        + "</div>"
        for name, score, setup, why in EXAMPLE_ROWS
    )
    trust = "".join(
        f'<div class="hsf-card"><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p></div>' for t, d in TRUST_POINTS
    )
    return (
        f'{_CSS}<div class="hsf-land">'
        f'<section class="hsf-sec" aria-labelledby="hsf-what"><h2 id="hsf-what">What HSF does</h2>'
        '<p>Thousands of symbols become one ranked short list, using market context, technical signals, '
        'unusual activity and ML-assisted ranking to help you decide what deserves research time.</p>'
        f'<div class="hsf-grid">{pillars}</div></section>'
        f'<section class="hsf-sec" aria-labelledby="hsf-ex"><h2 id="hsf-ex">What a result looks like: What HSF AI surfaces and why</h2>'
        f'<div class="hsf-ex">{rows}</div><p class="hsf-note">{html.escape(EXAMPLE_NOTE)}</p>'
        f'<p class="hsf-score">{html.escape(HSF_SCORE_ONE_LINE)} '
        f'<a href="{METHODOLOGY_HREF}" target="_self">How HSF Score works</a></p></section>'
        f'<section class="hsf-sec" aria-labelledby="hsf-trust"><h2 id="hsf-trust">Why you can trust what you see</h2>'
        f'<div class="hsf-grid">{trust}</div></section>'
        f'<section class="hsf-sec" aria-labelledby="hsf-plans"><h2 id="hsf-plans">Plans</h2>'
        f'{plans_html()}<p class="hsf-note">Start free — no credit card required.</p></section>'
        f'<p class="hsf-disc">{html.escape(DISCLAIMER)}</p>'
        '</div>'
    )


def render_signed_out_hero() -> None:
    """Hero above the sign-in form. Never raises."""
    if st is None:
        return
    try:
        track_landing_visit_once()
        st.markdown(hero_html(_logo_data_uri()), unsafe_allow_html=True)
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("landing hero", exc)


def render_signed_out_details() -> None:
    """Product details below the sign-in form. Never raises."""
    if st is None:
        return
    try:
        st.markdown(details_html(), unsafe_allow_html=True)
        st.page_link("pages/methodology.py", label="How HSF works (methodology)")
        st.page_link("pages/billing.py", label="Compare plans in detail")
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("landing details", exc)
