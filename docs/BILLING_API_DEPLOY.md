# Run billing on the paid API service

The Next.js UI stays separate. The existing API can run checkout, customer
portal and Stripe webhooks in process, using the same Stripe account, prices,
customers and Postgres database. No subscription or schema migration is needed.
This change does not change DNS or Render services automatically.

## Configure Render

Keep the API's repository root, build command `pip install -r api/requirements.txt`
and start command `python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
Use its paid instance. Copy these settings from the billing service to the API
without exposing them in source control or in the browser:

| Setting | Value |
| --- | --- |
| `BILLING_MODE` | `embedded` (default is `external` for rollback) |
| `STRIPE_SECRET_KEY` | Same Stripe account and live/test mode as before |
| `STRIPE_WEBHOOK_SECRET` | Signing secret for the existing webhook endpoint |
| `STRIPE_PRICE_PRO`, `STRIPE_PRICE_PREMIUM` | Existing monthly price IDs |
| `STRIPE_PRICE_PRO_YEARLY`, `STRIPE_PRICE_PREMIUM_YEARLY` | Existing yearly IDs, if offered |
| `STRIPE_PRICE_PRO_LEGACY`, `STRIPE_PRICE_PREMIUM_LEGACY` | Preserve every grandfathered price ID |
| `DATABASE_URL` | Same database currently used by billing and the API |
| `APP_SUCCESS_URL`, `APP_CANCEL_URL`, `APP_PORTAL_RETURN_URL` | `https://app.hsfinest.com` |
| `APP_BASE_URL` | `https://app.hsfinest.com` |
| `APP_RETURN_HOSTS` | `app.hsfinest.com,ai.hsfinest.com,hsf-web-v1.onrender.com` (keep previous hosts during transition) |
| `BILLING_WEBHOOK_URL` | `https://hsf-api-v1.onrender.com/billing/webhook` (use the actual API origin) |
| `BILLING_DEBUG_STATUS` | Leave disabled outside an authorized diagnostic window |

Preserve the API's existing JWT, email, database and worker settings. Billing's
mounted app does not start another alert worker. Keep `REALTIME_ALERTS_ENABLED`
disabled on the old billing service. Embedded billing validates required
configuration at startup; external mode does not import the Stripe service.

The production UI origin is `https://app.hsfinest.com`. Add that custom domain
to the Render UI service and verify DNS and HTTPS before changing return URLs.
Retain `ai.hsfinest.com` during transition for existing links. Set the UI's
`APP_BASE_URL` and the API's `APP_BASE_URL` to the new origin together so email
links and Stripe return links agree. GitHub jobs that send account emails also
need `APP_BASE_URL=https://app.hsfinest.com` in their environment.

## Cut over and verify

1. Deploy the updated API with the configuration above. Keep the old billing
   service running while verifying the new API's `/healthz` and `/billing/health`.
   The web UI continues using `/v1/billing/checkout` and `/v1/billing/portal`.
   These routes now invoke billing directly, without a network round trip or
   a second billing token. JWT authorization and email verification still apply.
2. Verify new checkout, existing-subscriber plan management, portal and cancel
   flow with appropriate test accounts. A Stripe/portal failure for an existing
   customer returns an error rather than creating a duplicate subscription.
3. In Stripe, **edit the existing webhook endpoint URL** to
   `https://hsf-api-v1.onrender.com/billing/webhook`. Retain its signing secret
   and enabled events: `checkout.session.completed`,
   `customer.subscription.updated`, `customer.subscription.deleted`.
   Do not leave two active endpoints delivering the same events. If creating a
   replacement endpoint instead, its new signing secret must be set on the API
   before enabling delivery, and the old endpoint must be disabled.
4. Send/replay a Stripe event and verify a successful delivery and the correct
   account tier in the shared database. Replay it again and verify it is skipped
   as already processed. Check Stripe's delivery history and API error logs.
   The handler verifies the original raw body and signature; the UI never sees
   the Stripe secret. Existing event records continue to be used.
5. Any remaining Streamlit/external callers can set `BILLING_API_BASE` to
   `https://hsf-api-v1.onrender.com/billing`. The token-authenticated checkout and
   portal endpoints remain available there. The Next.js UI needs no change.
6. After all callers and webhook deliveries are verified, suspend the separate
   billing service to stop its compute charges. Keep its settings for rollback.

The default `external` mode keeps the current billing URL and behavior. To roll
back, keep/restore the old service, restore the Stripe endpoint's old URL and
corresponding secret, and set API `BILLING_MODE=external`. Restore any classic
client's `BILLING_API_BASE` as well. A redeploy is required for mode/env changes.

Code tests use mocked Stripe and Postgres; they do not confirm a live payment,
Render configuration or Stripe delivery. Complete the live checks before
suspending the old service.

## Returning from Stripe on a Mac

Account and Plans & Pricing offer a Monthly / Yearly selector. The choice is
remembered for this browser session and passed to checkout or the plan-change
portal. Paid subscribers can also choose their current plan with a different
interval. Configure both yearly price IDs on the API and include the prices in
Stripe's portal configuration. Customers confirm the amount and any proration
in Stripe before the change takes effect.

macOS may open Stripe in Safari outside an installed HSF web app. Embedded API
sessions return to `/billing/return`, which explains how to switch back or open
the account in the browser. No Streamlit restore token is added to these web
URLs. An upgrade confirmation flow redirects after completion; the general
Stripe portal also has a return link.

The originating HSF window remembers a pending billing flow for up to two
hours and makes up to three checks when it becomes visible or gains focus.
Account also offers **Refresh plan**. These checks call authenticated
`POST /v1/billing/sync`, then reload `/v1/me`. The API reads subscriptions for
the account's stored Stripe customer, recognizes only configured price IDs,
and updates the local plan from Stripe's confirmed active/trialing subscription.
It does not create or modify a Stripe subscription. Unknown prices, multiple
active subscriptions, mismatched customers and Stripe failures never grant
access; unknown prices produce an actionable configuration error. Webhooks
remain necessary for ongoing renewals and cancellations.

If Stripe shows Premium while HSF still shows Pro, verify that the subscription's
actual price ID is in `STRIPE_PRICE_PREMIUM`, `STRIPE_PRICE_PREMIUM_YEARLY` or
`STRIPE_PRICE_PREMIUM_LEGACY` (uppercase). A product name or dollar amount alone
does not identify the plan. Also verify webhook deliveries and that the Stripe
key and prices are all in the same test/live mode. A portal marked **Test mode**
does not confirm a live payment.
