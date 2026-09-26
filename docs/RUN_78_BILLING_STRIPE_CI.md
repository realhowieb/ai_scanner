# Run 78: Billing / Stripe Contract in CI

## Architecture

```text
Signed-out visitor
  -> shared pricing (`ui.pricing`)
  -> sign-up/sign-in
  -> Billing page (`pages/billing.py`)
  -> checkout adapter (`ui.checkout`)
  -> billing service `/create-checkout-session`
  -> Stripe Checkout or Billing Portal
  -> signed `/webhook`
  -> `users.tier` + Stripe customer/subscription/price identifiers
  -> DB-first tier resolver (`auth.tier_sync`)
  -> deterministic entitlements (`ui.app_session`)
  -> paid feature gates
```

Stripe prices are configuration (`STRIPE_PRICE_PRO` and
`STRIPE_PRICE_PREMIUM`), not user input. Checkout accepts only `pro` or
`premium`. Existing subscribers are sent to the Stripe Billing Portal instead
of receiving a second subscription.

The supported webhook contract is deliberately narrow:

- `checkout.session.completed` establishes the initial paid tier.
- `customer.subscription.updated` applies a price change, immediately handles
  a cancelled status, or keeps the current tier when cancellation is merely
  scheduled for period end.
- `customer.subscription.deleted` returns the account to Basic.
- Other signed event types are acknowledged without changing entitlements.

The `users` table is the subscription-state source used by the application.
`stripe_processed_events` records Stripe event IDs for retry deduplication.
`users.username` currently serves as the normalized account email/customer
mapping key.

## Configuration Boundary

The billing service consumes these environment variables:

| Variable | Purpose |
|---|---|
| `STRIPE_SECRET_KEY` | Stripe service authentication |
| `STRIPE_WEBHOOK_SECRET` | Webhook signature verification |
| `STRIPE_PRICE_PRO` | Configured Pro recurring price |
| `STRIPE_PRICE_PREMIUM` | Configured Premium recurring price |
| `DATABASE_URL` | Subscription/user persistence |
| `APP_SUCCESS_URL` | Checkout return URL |
| `APP_CANCEL_URL` | Checkout cancellation URL |
| `APP_PORTAL_RETURN_URL` | Optional portal return URL |
| `BILLING_API_BASE` | Streamlit-to-billing-service endpoint |

No value is committed or printed by the Run 78 CI job.

## Existing Coverage

Before Run 78, `tests/test_billing_service.py` contained import/health checks,
four price-to-plan cases, and two invalid-signature cases. Those tests were
decorated to skip when FastAPI was unavailable. The normal `smoke` job
installed only `requirements-dev.txt`, so every billing-service test skipped.

Run 72 and Run 73 tests already protected shared pricing presentation and
entitlement packaging. Tier-enforcement tests already covered direct feature
gates. They did not exercise the path from checkout or a webhook to the stored
tier.

No existing deterministic test requires Stripe credentials, a live database,
or network access. Streamlit-only presentation cases remain optional in the
lightweight environment.

## New Coverage

Run 78 adds contract tests for:

- Pro and Premium checkout price selection, customer metadata, and return URLs.
- Rejection of `admin`, raw price IDs, empty plans, and nonexistent plans.
- Missing application users and database lookup failures before Stripe calls.
- Existing active subscribers being directed to the Billing Portal.
- Initial Pro and Premium grants from a verified configured Stripe price.
- Pro-to-Premium and Premium-to-Pro subscription changes.
- Scheduled cancellation retaining the paid tier until it becomes effective.
- Effective cancellation/deletion returning the account to Basic.
- Free discovery remaining available after cancellation while paid and Premium
  capabilities disappear.
- Unknown prices failing closed to Basic.
- Missing customer mappings, malformed events, invalid signatures, and failed
  database writes producing no paid grant.
- Duplicate event IDs being acknowledged without reapplying a tier change.
- The shared Free / Pro / Premium names, `$19/mo` and `$39/mo` presentation,
  configured Stripe mappings, and entitlement tiers staying aligned.

Stripe's Python module and psycopg2 are replaced in `sys.modules` before the
billing service is imported. Endpoint tests use FastAPI's in-process
`TestClient`; no socket, Stripe API, Postgres server, customer, subscription,
or charge is created.

## CI Configuration

`.github/workflows/smoke.yml` now has a dedicated `billing-contract` job on the
same pull-request, `dev`, `main`, and manual triggers as Smoke Checks. It
installs `requirements-billing-test.txt`, which contains the existing developer
test tools plus only FastAPI and HTTPX. It then imports both required packages
and runs:

```bash
python -m pytest \
  tests/test_billing_service.py \
  tests/test_run72_pricing.py \
  tests/test_run73_tier_differentiation.py \
  tests/test_tier_enforcement.py \
  tests/test_app_session.py \
  -v --tb=short
```

The explicit import step makes dependency setup failures visible. Billing
service cases cannot silently skip in this job because FastAPI is installed.
The larger smoke and dependency jobs remain unchanged.

## Security Verification

- CI uses obvious fake values only; production Stripe secrets and price IDs are
  neither required nor exposed.
- Stripe and database boundaries are mocked and no network access is used.
- Checkout accepts canonical plan names, never a caller-provided price ID.
- Unknown webhook prices map to Basic, not a paid or Admin tier.
- A missing user, invalid customer email, invalid signature, malformed event,
  or failed database write cannot grant paid access.
- Admin is absent from the customer pricing and checkout contract.
- Redirect overrides are accepted only for the configured application host.

## Packaging Verification

The canonical customer plans remain:

- Free: discovery, HSF Score, current market opportunities, Basic Stock
  Intelligence, and the existing Basic feature allowance.
- Pro: `$19/mo`, monitoring/investigation capabilities and Pro feature gates.
- Premium: `$39/mo`, Pro capabilities plus the advanced research/workflow
  feature gates established in Run 73.

Pricing cells continue to derive from `FEATURE_MIN_TIER` and
`ALERT_LIMIT_BY_TIER`; Run 78 does not create another product configuration.

## Known Limitations

- The service does not separately process `customer.subscription.created` or
  invoice/payment events; the production contract uses checkout completion and
  subscription update/deletion events.
- Event deduplication is sequential rather than a single transaction spanning
  the user update and processed-event insert. Retried updates are logically
  idempotent, but concurrent duplicate delivery is not serialized.
- Stripe events are not ordered by Stripe `created` time. A late older
  subscription update could overwrite a newer tier. This predates Run 78 and
  is documented rather than changed in this CI-hardening run.
- Live Stripe test-mode checkout remains an optional operational verification,
  not a deterministic CI dependency.

## Test Results

Local isolated billing environment (Python 3.12, no Streamlit, Stripe SDK, or
psycopg2 installed):

- Billing service contract: 26 passed, 0 failed, 0 skipped.
- Billing/pricing/tier gate: 85 passed, 0 failed, 1 Streamlit-only presentation
  test skipped.
- Full lightweight suite: 1,774 passed, 0 failed, 74 skipped, 178 subtests.
- Warnings: FastAPI/Starlette lifecycle compatibility deprecations only; no
  billing assertion or collection warning.

GitHub Actions remains the authoritative clean Linux/Python 3.13 execution
after push.

## Backlog Status

- P1-11: DONE (Run 76, scanner row ordering aligned with the HSF headline).
- P1-12: DONE (Run 77, visible Breakout alert score scale).
- P1-18: DONE (Run 75, one onboarding system).
- P1-19: DONE (Run 75, one Custom Scan entry point).
- P1-20: DONE (Run 75, Today landing after sign-in).
- P1-21: WAIT; dependency/security re-certification is not part of Run 78.
- P1-22: DONE when the new `billing-contract` CI job passes on the pushed
  commit.

The expected next product work is Run 79: P2-11 + P2-12 + P2-15 information
hierarchy and trust polish, unless CI or production evidence reveals a more
severe issue.
