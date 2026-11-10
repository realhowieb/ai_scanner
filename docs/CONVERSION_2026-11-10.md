# Conversion pack (2026-11-10)

Six changes aimed at the drop-off between landing and paying (PostHog, 14 days to
2026-11-10: 40 homepage visitors, 9 on /signup, 1 sign-up; 13 on /pricing, 2 checkout starts).

| # | Change | Where | Switch |
|---|--------|-------|--------|
| 1 | Paid plan buttons on Pricing go to `/signup?plan=pro|premium&interval=`; the plan is kept in the browser for a week (`hsf-plan-intent`). After email confirmation the verify page offers "Continue to Pro checkout", which opens `/account?upgrade=pro&interval=month#plan` and starts Stripe Checkout by itself. A reminder strip shows in the app until it's done or dismissed. | `web/src/lib/planIntent.ts`, `web/src/features/PlanIntent.tsx` | none (browser only) |
| 2 | Sign-up asks only email, password, the agreement and the Cloudflare check. The display name comes from the email (`jane.doe`, or `jane.doe-1234` when taken) and can be changed in Account. Refused sign-ups send a `signup_failed` funnel event with a reason tag (email, password, taken, captcha, network, server...). | `api/account.py`, `web/src/features/AuthForms.tsx` | none |
| 3 | One follow-up email for a checkout started 20-72 hours ago and not finished, with a link back to that plan's checkout and the Founding 100 seats left while on sale. At most once per account, ever; respects the "Tips and reminders" preference. | `services/checkout_followup.py` (daily account mail, after 15:00 UTC) | `HSF_CHECKOUT_FOLLOWUP_ENABLED=0` |
| 4 | Every new Starter account gets 7 days of free Pro when it confirms its email (no card; skipped when an invite reward applies, for admins, paid plans and founders; once per account). Uses the same free-Pro grants as invites, so the trial-ending emails and the Account notice already cover it. Pricing shows "Try Pro free for 7 days". | `api/referrals.py` `start_new_trial`, `GET /v1/plans` `trial_days` | `HSF_NEW_TRIAL_DAYS` (default 7, 0 = off) |
| 5 | Locked Day Trader, pre-market gappers and earnings cards show one real example and a count (`GET /v1/previews/{kind}`), from data the API already keeps warm. Cached 2 minutes; no new data fetches. | `api/previews.py`, `web/src/features/LockedPreview.tsx` | none |
| 6 | `sectors.csv` now also carries industry and market cap from the Nasdaq screener. Stock pages show market cap, size, sector and industry; the Scanner gets a Size filter (`cap=micro|small|mid|large|mega`). | `scripts/refresh_sectors.py`, `api/sectors.py` | none |

After merging into dev, dispatch **Refresh Universe** on dev once so `sectors.csv` gains the
new columns (api.nasdaq.com is blocked from the cloud sandbox). Until then stock pages show
only the sector and the Size filter stays hidden.

Memory: the new code reads cached data only (previews) or one small CSV already loaded
(~7,000 rows; the two extra columns add well under 1 MB).
