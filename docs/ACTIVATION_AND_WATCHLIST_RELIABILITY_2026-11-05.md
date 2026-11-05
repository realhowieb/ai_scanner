# Activation dashboard and watchlist recovery

## Saved PostHog dashboard

[HSF activation and usage](https://us.posthog.com/project/679220/dashboard/2241150)
was updated in the existing HSF project on November 5, 2026. This is a live
PostHog configuration change, separate from deployment of the watchlist fix.

- [Signup → stock review → saved stock](https://us.posthog.com/project/679220/insights/BqHA4KBw)
  replaces the previous signup → empty watchlist-created definition. It uses the
  existing signup_completed → stock_viewed → watchlist_stock_added events, ordered
  within seven days, over the last 30 days. Step percentages are relative to the
  previous step. This measures a successful stock addition, not list creation.
- [Saved-stock browsers: day 1–7 returns](https://us.posthog.com/project/679220/insights/f7hJSao7)
  is a native retention insight: first-ever watchlist_stock_added, returning on
  page_viewed, eight daily 24-hour bands (Day 0 through Day 7), non-cumulative,
  weighted mean, last 30 days. Day 0 is the initial period, not a later return.
  Day 7 is a separate 24-hour band, not a cumulative seven-day return rate.
  Recent cohorts have not had time to complete all bands.

The return population is all browsers with a first saved-stock event, not only
those completing the signup funnel. Use these as complementary views rather
than multiplying their rates into an end-to-end conversion percentage.

The existing research-to-alert, scanner/alert activity, journal and checkout
insights were retained. The new funnel is at the top and retention spans the
full dashboard width. Both native queries were executed successfully, and the
saved dashboard was verified visually.

These are exploratory definitions, not approved Data Catalog metrics (the
catalog had no entries). Existing internal/test-account inclusion was retained
and is stated on the dashboard. There is no new identification, event capture,
replay, or collection of account details. The anonymous identity is browser
based; sign-out, opt-out, storage resets and different devices can break the
journey. Very small cohorts should not be treated as stable conversion evidence.

## Watchlist performance

The How your stocks did card now distinguishes loading, a successful response
without price history, and a failed request. Request failures use the existing
error and retry component, including its Retry-After handling. If a background
refresh fails, the last loaded table stays visible with a stale-results notice.

Regression coverage checks loading visibility, empty-history messaging,
503 → Retry → successful table, and preserving a table after a failed refresh.
No API contract, scoring, or additional analytics collection changed.
