# Shared application rules

`domain` owns pure plan limits, entitlements, plan names, password policy and
upgrade explanations. API authorization and Streamlit session adapters import
the same rules. These modules import without a UI runtime or database.

`services.billing` and `services.account_tokens` own checkout requests and
single-use billing/restore tokens. The former `ui` module paths remain aliases
to the canonical implementations so existing integrations keep working.

With `BILLING_MODE=embedded`, the API invokes `billing_service.main` directly
for its authenticated checkout and portal routes and mounts Stripe delivery at
`/billing/webhook`. External mode remains available for rollout and rollback.
See [billing deployment](BILLING_API_DEPLOY.md) for settings and cutover checks.

`services.paper_orders` owns recoverable paper-order submission and journal
reconciliation for both frontends. `db.paper_orders` records confirmation
intents and pending broker results before an order can be submitted. No trading
credentials are stored in this table. The existing runtime schema initialization
creates the table and adds a journal lookup index; no production migration has
been run as part of this change.

Paper reconciliation is performed when Journal or paper activity is refreshed.
It does not submit orders. It uses a bounded batch of pending intents and only
imports confirmed fills, using a transaction lock to avoid duplicate journal
rows across workers. Existing journal duplicates are preserved rather than
silently deleted. Rotated credentials can recover an order only when the broker
confirms its client order ID and payload; an old confirmation is never submitted
against a newly connected account.

Repository-wide source checks use Git's tracked Python inventory. Local nested
checkouts and untracked copies are excluded. New source files enter that
inventory when they are added to Git.
