# Daily research workflows

Scanner and Custom scan now offer named account views (up to 20 each). Saving the same name replaces that view. A Scanner view stores the setup, earnings window, minimum score, price band, relative-volume floor, sort and page size. Views saved before the price band and relative-volume fields existed load as "Any". Views store settings only, never results; applying a view does not start a scan. Existing server plan checks and watchlist ownership still apply when the user runs it. Display preferences choose optional columns and compact spacing, across current scan tables and mobile cards. Ticker and score remain visible.

Pro and Premium can download the Scanner's current results as CSV (`can_export_csv`). The file holds every row the current filters leave, up to the plan's result cap, fetched in pages of 200 from `GET /v1/scans/latest`; it stops with a message if a new scan finishes mid-download rather than mixing two scans. Prices in the file are the scan's prices with the scan time in each row, not live quotes. Cells that start with `=`, `+`, `-` or `@` (other than plain numbers) are prefixed with an apostrophe so a spreadsheet never runs them.

Watchlist prices come from `GET /v1/quotes`. Each quote carries `source`: `trade` (the latest trade, shown with its time) or `daily_bar` (no trade was returned, so the price is that session's daily close, shown as "daily close"). The API re-reads quotes about once a minute, but on a weekend or holiday the latest trade itself can be days old, which is why each row shows its own time.

The shared search accepts company names and tickers, supports keyboard selection, and remembers six recent tickers in this tab, scoped by account. Company suggestions come from the existing Alpaca active tradable assets directory (six-hour process cache, five-second refresh timeout). A failed refresh retains the previous directory and retries after a minute. Without suggestions, exact ticker navigation remains available.

Select two to four names in the Scanner or visit `/stocks/compare`. Each stock uses the existing authenticated detail API and its plan redaction. Each column shows its own scan time, historical-score warning, score, signals, price, relative volume, reasons, risks, and watch items. An unavailable column has an independent retry; it does not hide the other stocks. Comparison scrolls horizontally on phones.

## Notifications

Alerts shows recent rule email statuses and browser push attempts separately. A push status of `accepted` confirms acceptance by the push provider, not display or reading on the device. `gone` means that browser subscription is no longer usable. `unconfirmed` means the network outcome was ambiguous and deliberately cannot be retried. `sending`/`retrying` left by a worker restart also stay unconfirmed by this system rather than risking duplicate delivery.

Manual retries atomically claim one failed receipt owned by the current account, target only its original registered device, check current entitlement and quiet hours, expire after four hours, and allow at most three total attempts. Accepted receipts cannot be retried. The ledger contains no tokens or endpoints; these remain in the existing device table. Old receipts for an account are removed when the next delivery begins after seven days. Account deletion and export include the new personal data.

Quiet hours use an IANA time zone and follow its daylight-saving rules. They suppress browser pushes and alert emails (scheduled alerts, rule alerts, and the standalone real-time worker), while in-app events continue. Suppressed notifications are not queued for later. Equal start/end disables the interval. The preference read follows the existing email preference behavior during database outages: original alerts can still be sent if preferences cannot be read.

## API and operations

- `GET/PATCH /v1/me/research-preferences`: authenticated, validated account settings. PATCH atomically merges supplied sections so updating table display cannot overwrite quiet hours. Updating the same section in two clients is last-write-wins.
- `GET /v1/symbols/search?q=...`: authenticated, bounded company/ticker suggestions.
- `GET /v1/alerts/deliveries`: the account's latest 50 browser receipts.
- `POST /v1/alerts/deliveries/{id}/retry`: account-owned, bounded manual retry.
- `GET /v1/admin/requests?q=...&problems_only=true`: admins only. Bounded process-local request ring, provider checks, and database-backed shared scan queue counts. Existing `/v1/admin/operations` remains unchanged.

Admin → Health → API operations searches support codes, route templates, and HTTP statuses. Requests taking at least one second are marked slow. It stores only request ID, method, route template, status, duration, and timestamp: no user identifiers, query strings, request/response bodies, headers, or tokens. The panel explicitly reports that it covers this API process since restart (up to 1,000 requests); it is not a durable cross-instance log archive. Services probes also populate recent provider checks.

The two additive tables (`hsf_workspace_preferences`, `hsf_push_deliveries`) are initialized through the repository's normal once-per-database schema helpers, with transaction advisory locks for concurrent first use and foreign keys that remove rows when an account is deleted. No production database commands are needed. API writes/search/retries and admin reads have separate request limits.

## Validation

Workflow tests cover account separation, schema validation, local-time/DST boundaries, quiet-hour suppression, and retry authorization. The isolated PostgreSQL CI job additionally checks section merges and simultaneous retries with real database transactions. Web tests cover saved view reload/deletion, search selection/history, table display, bounded comparison, partial failures, quiet-hour saving, delivery retries, and operations rendering.

Billing lifecycle coverage remains in the billing contract and PostgreSQL suites. Public production health, billing return, authenticated-route rejection, and unsigned webhook rejection can be checked without changing a subscription. Full signed-in Stripe sandbox checkout/renewal/cancellation verification requires an authenticated sandbox session and test events; those results must not be inferred from public health checks.
