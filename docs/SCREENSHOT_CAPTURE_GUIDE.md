# HSF Screenshot Capture Guide

## Setup

Set `HSF_SCREENSHOT_MODE=true` in the capture environment and restart Streamlit.
Use an authenticated private-beta account with real current data. Do not add or
edit records solely for a screenshot. Confirm the visible timestamp and market
session before capture. Screenshot mode defaults off when the variable is absent.

Recommended base viewport: **1600 x 1000**. Use **1920 x 1080** for Day Trader
and **1440 x 900** only for card-focused views. Keep browser zoom at 100% and
the sidebar collapsed.

## Scanner

- Route: `app.py`, Latest Scan Results, `Top ranked`.
- Viewport: 1600 x 1000.
- Visible: ticker, price, move, HSF Score, status, movement, setup, signals,
  PreBreakout, Breakout, and one selected explanation.
- Hide/crop: scan controls, raw all-column output, diagnostics, exports, chat.
- Headline: **Scan the market. Find the setups.**
- Caption: HSF evaluates the broader U.S. market and prioritizes the names that
  deserve attention, with each setup's evidence kept visible.

## Stock Intelligence

- Route: `pages/stock.py`, opened from a real ranked scanner result.
- Viewport: 1440 x 900 or 1600 x 1000.
- Visible: symbol, direction, lifecycle, Opportunity Score, PreBreakout, ML
  probability, Alert Priority, confirmations, caution factors, timestamp.
- Hide/crop: lower chart/history if it pushes current state below the fold.
- Headline: **Know why a ticker matters.**
- Caption: One canonical view connects HSF's current setup, supporting evidence,
  cautions, lifecycle, and model context.

## Market Brief

- Route: `pages/brief.py`.
- Viewport: 1600 x 1000.
- Market session: regular session or shortly after a healthy scheduled scan.
- Visible: market context, strongest current opportunities, meaningful changes,
  watchlist matches, and freshness.
- Hide/crop: orientation copy, methodology, empty secondary sections.
- Headline: **Start with what matters now.**
- Caption: Market context and prioritized opportunities in one concise brief.

## Day Trader

- Route: `pages/day_trader.py`.
- Viewport: 1920 x 1080.
- Filters: choose a real source with enough populated daily history.
- Visible: Ticker, Last, Change, Gap, Direction, DT Score, Setup, ADX, vs VWAP,
  RVOL, SuperTrend, EWO, and Volume.
- Hide/crop: auto-refresh, notification threshold, debug/detail expanders.
- Headline: **See intraday structure in context.**
- Caption: Live price action, trend, participation, and momentum signals together.

## My Watchlist

- Route: `pages/watchlists.py`, `Attention` view.
- Viewport: 1440 x 900.
- Visible: tracked/attention/strengthening/fading summary and 3–5 attention cards.
- Hide/crop: watchlist management and bulk actions below the intelligence view.
- Headline: **Your market, prioritized.**
- Caption: HSF tracks the names you care about and surfaces meaningful changes.

## Supporting captures

- PreBreakout: 1600 x 1000, latest scan, ranked table with at least five real
  candidates. Headline: **Find developing setups earlier.**
- Alerts: 1440 x 900, crop to Recent Intelligence. Headline: **Know when the
  setup changes.**
- Historical Replay: 1600 x 1000 only when one symbol has a full session with
  multiple lifecycle events and matured outcomes.

## Capture checks

1. Confirm the visible data is real and the timestamp is honest.
2. Confirm no debug text, email, internal ID, or browser credential is visible.
3. Confirm unknown values render as `—` or unavailable.
4. Confirm candidate order matches normal mode.
5. Capture PNG at native resolution; crop browser chrome only.

