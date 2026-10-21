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
  allowed browser privacy settings. The wizard configured a local ignored environment file; production configuration is unverified.
- Explicit events cover page categories, Scanner/stock research, successful
  watchlist and alert actions, accepted custom scans, journal actions, checkout starts and signup completion.
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

- 48 frontend test files / 379 tests passed, including initialization ordering, optional startup, metadata allowlists and log failure isolation.
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
Do not supply a personal API key. The wizard created a [usage dashboard](https://us.posthog.com/project/679220/dashboard/2241150) and [setup notebook](https://us.posthog.com/project/679220/notebooks/9c8820e560fe).
`web/README.md` documents the final events and suggested funnel/retention reports.
Cross-device attribution and session replay are intentionally unavailable.


## Wizard reconciliation

The generated setup was reviewed before commit. Restored consent-gated SDK
initialization, anonymous identifiers, the outbound property allowlist, optional
startup without keys and the existing error reporter/CSP. Email identification,
automatic exception capture and broader script/worker permissions were not retained.
The browser instrumentation hook shares the original ordered initialization promise.

Operational method/status logs are available only with `POSTHOG_LOGS_ENABLED=1`
(default off), with best-effort export and no global provider registration. This
server setting is independent of the browser usage preference. Tests disable
analytics globally to prevent local project credentials from sending test activity.
The local wizard helper files remain on disk and are excluded from the PR.

Live production delivery, deployment environment settings, upstream AI observability
and error source-map uploads remain unverified/outside this change. The wizard
notebook describes its generated snapshot; this document describes the reviewed
implementation. No production deployment was performed.
