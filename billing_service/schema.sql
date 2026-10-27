-- Additive, repeatable billing schema. No Stripe secrets or raw events are stored.
CREATE TABLE IF NOT EXISTS stripe_processed_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS hsf_billing_state (
    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
    snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    checkout_intent JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_sync_at TIMESTAMPTZ,
    last_sync_error TEXT
);
CREATE TABLE IF NOT EXISTS hsf_billing_failures (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    error_code TEXT NOT NULL,
    failed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
