# Lightweight Charts and PostHog — October 21, 2026

Implemented locally on the branch updated to main `fc82314`. Existing local UI
improvements remain intact. No scoring, ranking, outcome, entitlement, billing,
or authentication decisions changed. No deployment or account creation occurred.

## Stock chart

- Added browser-loaded Lightweight Charts 5.2.1 with candles, a volume histogram,
  EMA 9/21 and SMA 20/50, HSF signal and cross markers, and upcoming earnings dates.
- Kept range controls, IEX volume explanations, saved-price wording, keyboard
  session readouts and the SVG fallback. Pan and zoom do not require a new feed.
- Normalize duplicate/out-of-order dates and incomplete OHLC only in chart input;
  source bars and research calculations are untouched.
- Preserve zoom across overlay changes, resize with the container and release
  observers/chart resources on unmount. Vertical phone swipes can scroll the page.
- Retain TradingView attribution and serve its notice/license with the app.

## Usage analytics

- PostHog JS 1.438.13 initializes only with a configured public project key and
  allowed browser privacy settings. No real project key has been configured.
- Explicit events cover page categories, Scanner/stock research, successful
  watchlist actions, alert creation and signup completion.
- No identity calls, account fields, symbols, watchlist names, notes, form values,
  raw URLs, query strings or referrers are sent. An outbound allowlist removes SDK
  enrichment and top-level person properties. Anonymous identifiers support
  browser-level retention; sign-out/session expiry resets them.
- Automatic capture, replay, heatmaps, error capture, surveys and feature flags
  are disabled. Account has a browser-local opt-out. DNT/GPC take priority.
- Analytics errors do not delay or fail product actions. The default blank key
  causes no SDK initialization or external analytics requests. CSP permits only
  the chosen US or EU ingestion origin when configured.

## Verification

- 46 frontend test files / 362 tests passed.
- Type checking, lint and production build passed.
- Desktop 1440×1000 and phone 390×844 local browser checks passed: Canvas rendered,
  keyboard session selection, wheel zoom, range selection, overlay controls, no
  horizontal overflow, no browser errors.
- Real SDK browser checks used a dummy project key and intercepted all external
  requests. The decoded events contained `/stocks/:ticker`, excluded test symbols,
  account-like strings and query tokens, and stopped after opting out. No CSP or
  browser errors occurred. A normal visitor was simulated because PostHog filters
  automation traffic by default; the production bot filter stays enabled.
- Synthetic fixture service rejects data mutations. No real accounts, watchlists,
  alerts, email delivery or trades were touched.

Screenshots are included in the PR under `docs/previews/charts-analytics-2026-10-21/`:

- [Desktop chart](previews/charts-analytics-2026-10-21/stock-chart-desktop.png)
- [Phone chart](previews/charts-analytics-2026-10-21/stock-chart-phone.png)
- [Analytics privacy setting](previews/charts-analytics-2026-10-21/analytics-privacy-desktop.png)

The local gallery and compact browser results remain under
`artifacts/charts-analytics-2026-10-21/`.

## Activation

Set `NEXT_PUBLIC_POSTHOG_KEY` to the public project key and
`NEXT_PUBLIC_POSTHOG_HOST` to the US or EU ingestion host, then rebuild the web app.
Do not supply a personal API key. No PostHog dashboards have been provisioned;
`web/README.md` documents the events and suggested funnel/retention reports.
Cross-device attribution and session replay are intentionally unavailable.
