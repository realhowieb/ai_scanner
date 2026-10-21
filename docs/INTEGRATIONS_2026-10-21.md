# Integration setup and verification — October 21, 2026

## Where each key belongs

Never commit keys or paste them into chat. The PostHog **project key** is the public
capture token; a personal API key is not needed by the running app.

| Integration | Hosting service | Environment settings |
| --- | --- | --- |
| Browser analytics | Render `hsf-web-v1` (or `hsf-web-beta`) | `NEXT_PUBLIC_POSTHOG_KEY`, `NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com` |
| AI operational metrics | Python HSF API service | `POSTHOG_AI_ENABLED=1`, `POSTHOG_PROJECT_KEY`, `POSTHOG_HOST=https://us.i.posthog.com` |
| Error monitoring | Python HSF API service | `SENTRY_DSN`, `SENTRY_ENV=production`, `SENTRY_RELEASE` (deployed commit) |
| Browser push | Python HSF API service | Existing `VAPID_PRIVATE_KEY`, `VAPID_SUBJECT`, `HSF_WEB_PUSH_ENABLED=1` |

For EU PostHog projects use `https://eu.i.posthog.com` for the relevant host.
Browser variables are compiled into the Next.js build: saving them requires a
web rebuild/redeploy before the live browser can use them. Server variables
require restarting/redeploying the API. Nothing in this change deploys services.
Local browser variables belong in ignored `web/.env.local`. API variables belong
in the API process environment (for example an ignored local environment file
loaded by your development runner). These are independent environments.

## Browser analytics

The existing anonymous event collection remains unchanged: explicit events,
route templates, a final property allowlist, browser opt-out, DNT/GPC, no identity,
replay or automatic exceptions. Use the existing [dashboard](https://us.posthog.com/project/679220/dashboard/2241150).
The saved **Research to alert activation** insight now tracks `scanner_viewed` → `stock_viewed` →
`watchlist_stock_added` → `alert_created` within 14 days; weekly repeat `scanner_viewed` visits
measure browser retention, not account retention. Avoid a scanner-to-signup
conversion interpretation: signup may precede authenticated Scanner access.

Read-only production checks during this task found web/API liveness healthy,
no PostHog ingestion origin in the live web CSP, and a PostHog project banner
saying it has no events. The charts displayed sample data, not observed usage.
Those checks do not prove hosting environment settings or a deployed commit.
After setting the web variables and authorizing a rebuild, verify actual events
in PostHog. Keep synthetic test traffic in a separate development project.

## Browser push

Push is already implemented across service worker, subscription routes, encrypted
sending, delivery records, quiet hours, and existing price/setup/HSF rules. This
change improves the existing integration rather than adding another sender:

- Preserve an existing subscription when an API registration refresh fails.
- Roll back a new subscription if the server cannot register it.
- Keep the UI enabled and show a useful error when unsubscribe fails.
- Refresh browser permission state when returning from site settings.
- Explain per-browser opt-in, quiet hours, and device delivery limitations.
- Constrain both pushed and clicked notification links to the app's origin.

Users enable this themselves under Alerts → Delivery. Pro/Premium/admin access
is unchanged. On iPhone/iPad, install to the Home Screen first. No permission is
requested on page load. Delivery acceptance is not proof a device displayed or
read a notification. Existing quiet-hour skipping and retry semantics are unchanged.

Generate a VAPID pair only if this service has never had one, using
`python scripts/generate_vapid_keys.py` in a private terminal. Reuse an existing
key; rotating it invalidates browser subscriptions. No keys were generated or
rotated here, no production subscription was created, and no push was sent.

## Sentry

The API already forwards bounded web crash reports and captures backend errors.
The integration is verified with a real Sentry SDK and an in-memory transport:
synthetic browser and API errors both arrive without external delivery. Missing
DSN remains a no-op. SDK configuration explicitly disables default PII, local
variables and request bodies; tracing stays off. No second browser error collector
was added. This does not establish production delivery or provide browser source-map
symbolication. Configure the API DSN/release and verify a controlled incident after
an authorized deployment; don't send real user content as test errors.

## AI observability

The two shared Anthropic call paths now emit optional `$ai_generation` metadata:
model, provider, coarse feature, duration, input/output/cache token counts, and
coarse provider error category. Each call has fresh random IDs. There are no
account identifiers, tickers, prompts, completions, message histories, URLs or raw
exception messages. PostHog can estimate costs for recognized model/token data;
these estimates are not the billing ledger. Shared cached answers produce no new
model-call event, and pre-call limit/kill-switch rejections are not provider attempts.

Metrics use the documented [capture API](https://posthog.com/docs/api/capture) and
[generation fields](https://posthog.com/docs/ai-observability/generations) with the
existing requests library. No provider wrapper, dependency upgrade or new backend
service is required. A single lazy daemon worker drains a bounded queue of 64
records with short timeouts and no redirects/retries. Failure, overload and process
shutdown can drop telemetry; they never change AI answers, counting or limits.
Only US/EU PostHog hosts are accepted. This is server operational telemetry and is
independent of the browser usage preference, like existing server error monitoring.
The default is off. No real AI request or PostHog generation event was sent here.

## Read-only readiness command

Run in each hosting service's environment:

```sh
python scripts/integration_readiness.py
```

From the repository, optionally check public liveness and the web analytics CSP:

```sh
python scripts/integration_readiness.py --web-url https://app.hsfinest.com --api-url https://hsf-api-v1.onrender.com
```

Output contains configuration-presence booleans and status codes, never values.
Local results describe the current process environment, not Render's settings.
A healthy endpoint or configured key is not proof of delivery. The command
never sends analytics, errors, notifications, emails, AI requests, or trades.

## Verification

- Backend integration/guardrail tests: 88 passed, 1 skipped, 4 subtests passed.
- Frontend: 49 test files / 388 tests passed; type checking, lint and production build passed.
- CI runs the backend notification/monitoring suite with actual API dependencies.
- No scoring, rankings, outcome calculations, authentication, billing, plan access,
  user watchlists or existing alert rules changed.

- Desktop (1280 px) and phone (390 px) local browser checks: no horizontal overflow;
  alert tabs support arrow keys and Tab reaches the described push checkbox.
- [Desktop notification controls](previews/integrations-2026-10-21/notifications-desktop.jpg)
- [Phone notification controls](previews/integrations-2026-10-21/notifications-phone.jpg)
- [Saved activation funnel](previews/integrations-2026-10-21/posthog-activation-funnel.jpg),
  displaying explicitly labeled sample data because the project has no events yet.
