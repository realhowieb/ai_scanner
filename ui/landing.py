"""Run 62 — signed-out landing experience.

Replaces the old "giant logo + login form" first screen with:

  hero (small logo, product name, tagline, one-paragraph description)
  → three real product views (Scanner, Stair-Stepper, Market Brief)
  → sign-in / create-account form (unchanged auth logic, in ui.auth)
  → What HSF does · why you can trust it · disclaimer

All copy comes from ui.product_copy (factual, no performance claims). Product
screenshots are current customer-facing views captured without account,
browser, admin, or diagnostic chrome.
"""
from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

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

SHOWCASE_VIEWS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    (
        "scanner",
        "SCANNER",
        "Ranked Opportunities",
        "Thousands of symbols. One focused shortlist.",
        "assets/scanner-showcase.webp",
        "HSF Scanner showing ranked opportunities, HSF Scores, statuses, and why the leading setup ranked",
    ),
    (
        "stair-stepper",
        "DAY TRADE",
        "Stair-Stepper",
        "Analyze smooth intraday momentum across multiple confirmation windows.",
        "assets/day-trader-stair-stepper.webp",
        "HSF Day Trader Stair-Stepper controls and ranked one-minute trend results",
    ),
    (
        "market-brief",
        "MARKET INTELLIGENCE",
        "Market Brief",
        "Know the market environment before evaluating the setup.",
        "assets/market-brief-showcase.webp",
        "HSF Market Brief showing market regime, index performance, breadth, and intelligent alerts",
    ),
)
# Streamlit serves pages/methodology.py at /methodology.
METHODOLOGY_HREF = "/methodology"

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
.hsf-cta-secondary{display:inline-block;color:inherit!important;padding:8px 4px;text-decoration:none;font-weight:650}
.hsf-cta-secondary:hover{text-decoration:underline}
.hsf-cta-hint{font-size:.9rem;opacity:.8;margin:0}
.hsf-showcase{margin:16px 0 22px;border:1px solid var(--hsf-line);border-radius:8px;
  background:#0d1016;overflow:hidden;box-shadow:0 14px 40px rgba(0,0,0,.22)}
.hsf-showcase-head{display:flex;align-items:end;justify-content:space-between;gap:12px;padding:14px 16px 10px}
.hsf-showcase-head h2{font-size:1.15rem;margin:0;padding:0}
.hsf-showcase-kicker{font-size:.72rem;font-weight:750;letter-spacing:.08em;color:#5bd17d;white-space:nowrap}
.hsf-show-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}
.hsf-show-control{position:absolute;width:1px;height:1px;opacity:0}
.hsf-show-tab{display:flex;align-items:center;justify-content:center;min-height:46px;padding:8px 10px;
  border-top:1px solid var(--hsf-line);border-bottom:1px solid var(--hsf-line);cursor:pointer;
  font-size:.88rem;font-weight:650;text-align:center;opacity:.68;background:rgba(255,255,255,.015)}
.hsf-show-tab+.hsf-show-control+.hsf-show-tab{border-left:1px solid var(--hsf-line)}
.hsf-show-control:focus-visible+.hsf-show-tab{outline:2px solid #5bd17d;outline-offset:-3px}
#hsf-view-scanner:checked+.hsf-show-tab,
#hsf-view-stair-stepper:checked+.hsf-show-tab,
#hsf-view-market-brief:checked+.hsf-show-tab{opacity:1;color:#fff;background:rgba(91,209,125,.09);
  box-shadow:inset 0 -3px 0 #5bd17d}
.hsf-show-panels{grid-column:1/-1;min-width:0}
.hsf-show-panel{display:none;margin:0}
#hsf-view-scanner:checked~.hsf-show-panels .hsf-panel-scanner,
#hsf-view-stair-stepper:checked~.hsf-show-panels .hsf-panel-stair-stepper,
#hsf-view-market-brief:checked~.hsf-show-panels .hsf-panel-market-brief{display:block}
.hsf-show-copy{padding:13px 16px 11px}
.hsf-show-copy p{margin:0}
.hsf-show-eyebrow{font-size:.7rem;font-weight:750;letter-spacing:.07em;color:#5bd17d}
.hsf-show-title{font-size:1.02rem;font-weight:700;margin-top:2px!important}
.hsf-show-desc{font-size:.88rem;opacity:.76;margin-top:2px!important}
.hsf-show-media{aspect-ratio:1.56/1;overflow:hidden;background:#0d1016;border-top:1px solid var(--hsf-line)}
.hsf-show-media img{display:block;width:100%;height:100%;object-fit:cover;object-position:top left}
.hsf-show-missing{display:grid;place-items:center;height:100%;min-height:260px;opacity:.72}
.hsf-sec h2{font-size:1.2rem;margin:22px 0 10px;padding:0}
.hsf-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.hsf-card{border:1px solid var(--hsf-line);border-radius:10px;padding:12px 14px;background:var(--hsf-soft)}
.hsf-card h3{font-size:1rem;margin:0 0 4px;padding:0}
.hsf-card p{margin:0;font-size:.9rem;opacity:.85;line-height:1.4}
.hsf-note{font-size:.8rem;opacity:.7;margin:6px 0 0}
.hsf-score{font-size:.95rem;margin:10px 0 0;max-width:62ch}
.hsf-disc{font-size:.85rem;opacity:.8;margin:18px 0 4px}
@media (max-width:640px){.hsf-hero{margin-top:4px}}  /* phones: the page padding already clears the header */
@media (max-width:520px){
  .hsf-hero{gap:12px;align-items:flex-start}
  .hsf-hero img{width:64px;margin-top:4px}
  .hsf-showcase-head{display:block;padding:12px 12px 9px}
  .hsf-showcase-kicker{display:block;margin-bottom:3px}
  .hsf-show-tab{min-height:48px;padding:7px 5px;font-size:.78rem}
  .hsf-show-copy{padding:11px 12px 9px}
  .hsf-show-media{aspect-ratio:1.15/1}
  .hsf-show-media img{width:175%;max-width:none;height:auto;min-height:100%;object-fit:cover}
  .hsf-panel-scanner .hsf-show-media img{margin-left:-2%}
  .hsf-panel-stair-stepper .hsf-show-media img{margin-left:-38%}
  .hsf-panel-market-brief .hsf-show-media img{margin-left:-3%}
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


@lru_cache(maxsize=8)
def _asset_data_uri(relative_path: str) -> str:
    """Return a packaged landing asset as a data URI without exposing a file path."""
    try:
        path = ROOT / relative_path
        mime = "image/webp" if path.suffix.lower() == ".webp" else "image/png"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
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
        '<a class="hsf-cta-secondary" href="#hsf-product-showcase">Explore the platform ↓</a>'
        '<p class="hsf-cta-hint">Sign in or create a free account. No credit card required.</p></div>'
        '</div></div></div>'
    )


def showcase_html(asset_uris: Dict[str, str] | None = None) -> str:
    """Three authentic, lightweight product views with CSS-only tab controls."""
    sources = asset_uris or {slug: _asset_data_uri(path) for slug, _k, _t, _d, path, _a in SHOWCASE_VIEWS}
    controls = []
    panels = []
    for index, (slug, eyebrow, title, description, _path, alt) in enumerate(SHOWCASE_VIEWS):
        checked = " checked" if index == 0 else ""
        controls.append(
            f'<input class="hsf-show-control" type="radio" name="hsf-product-view" '
            f'id="hsf-view-{slug}"{checked}>'
            f'<label class="hsf-show-tab" for="hsf-view-{slug}">{html.escape(title)}</label>'
        )
        source = sources.get(slug, "")
        loading = "eager" if index == 0 else "lazy"
        priority = ' fetchpriority="high"' if index == 0 else ""
        media = (
            f'<img src="{source}" alt="{html.escape(alt)}" loading="{loading}" '
            f'decoding="async"{priority}>'
            if source
            else '<div class="hsf-show-missing">Product preview unavailable.</div>'
        )
        panels.append(
            f'<figure class="hsf-show-panel hsf-panel-{slug}">'
            '<figcaption class="hsf-show-copy">'
            f'<p class="hsf-show-eyebrow">{html.escape(eyebrow)}</p>'
            f'<p class="hsf-show-title">{html.escape(title)}</p>'
            f'<p class="hsf-show-desc">{html.escape(description)}</p>'
            '</figcaption>'
            f'<div class="hsf-show-media">{media}</div>'
            '</figure>'
        )
    return (
        f'{_CSS}<section class="hsf-showcase" id="hsf-product-showcase" '
        'aria-labelledby="hsf-showcase-title">'
        '<div class="hsf-showcase-head">'
        '<span class="hsf-showcase-kicker">ACTUAL PRODUCT VIEW</span>'
        '<h2 id="hsf-showcase-title">What HSF AI surfaces and why</h2>'
        '</div>'
        '<div class="hsf-show-grid" role="radiogroup" aria-label="Product views">'
        + "".join(controls)
        + '<div class="hsf-show-panels">'
        + "".join(panels)
        + '</div></div></section>'
    )


def details_html() -> str:
    pillars = "".join(
        f'<div class="hsf-card"><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p></div>' for t, d in PILLARS
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
        f'<section class="hsf-sec" aria-labelledby="hsf-score"><h2 id="hsf-score">One clear opportunity score</h2>'
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
        st.markdown(showcase_html(), unsafe_allow_html=True)
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
