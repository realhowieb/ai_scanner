# Provider failover and Neon restore (P2-43)

What to do when an outside service fails, and how to restore the live database.
Short on purpose: find the failing service, read its row, do the first step.

Where things run:

| Piece | Runs on | Deploys from |
|---|---|---|
| Streamlit app | Streamlit Community Cloud | `dev` |
| Billing service + live price alerts | Render (`billing_service/`) | `dev` (auto) |
| Scheduled scans, digest, wrap, alerts | GitHub Actions (`scheduled-scans.yml`) | `main` |
| Scan scheduler | cron-job.org → `workflow_dispatch` on `main` | — |
| Database | Neon, branch `live` (default branch) | — |

Secrets live in three separate places (Streamlit Cloud secrets, GitHub repository
secrets, Render environment). Changing a key in one does not change the others.

## First look

1. **Admin page → System health, Email delivery, Email setup cards.** One status
   per area; the Email setup card shows all three environments.
2. **Sentry** for new errors (`component=cron` is the scheduled job).
3. **GitHub → Actions → Scheduled Scans**: did the last run start, and did it pass?
4. **Billing:** `https://ai-scanner-h2c8.onrender.com/health` (`ok`, `db`, `email`).

## Provider by provider

### Market data: Alpaca

- **Symptom:** scans finish with far fewer rows, coverage drops on System health,
  or logs show Alpaca timeouts / 401 / 403.
- **Built in:** each batch that times out is recorded as a provider failure (not a
  filter skip). Symbols Alpaca doesn't return fall back to yfinance, unless
  `PRICE_SKIP_YF_FALLBACK=1` (the scheduled scan sets it for speed).
- **Short outage:** do nothing. The next cron slot (hourly) rescans.
- **Longer outage:** check status.alpaca.markets. To lean on yfinance for a manual
  run, dispatch the scan with the fallback on (unset `PRICE_SKIP_YF_FALLBACK` for
  that run). yfinance is slower and rate-limited, so expect partial coverage.
- **401/403:** the key was revoked or rotated. Update `ALPACA_API_KEY_ID` /
  `ALPACA_API_SECRET_KEY` in all three secret stores.
- Do not change scanner logic to work around an outage (scanner core is frozen).

### Email: Resend (SMTP)

- **Symptom:** Email delivery card WARNING, Sentry "send(s) failed", or users say
  verification / reset emails don't arrive. Admins see the reason on the
  resend-verification box (not configured, login rejected, sender rejected,
  recipient refused, network, provider error).
- **Built in:** the morning digest and evening wrap are marked sent for the day only
  after at least one send succeeds, so a full outage retries at the next cron slot
  inside the send window (digest 6 AM–noon ET, wrap from 4 PM ET).
- **Login rejected:** the API key was revoked. Create a new *sending-only* key in
  Resend and update `SMTP_PASS` on Streamlit Cloud, GitHub and Render.
- **Sender rejected:** `SMTP_FROM` must be on the verified domain
  (`ai.hsfinest.com`). Check the domain is still Verified in Resend (DNS changes at
  the registrar can un-verify it).
- **Recipient refused (only the owner receives):** the account fell back to test
  mode or the domain lost verification.
- **Resend down for hours:** switching providers means new SMTP settings in all three
  stores plus sender-domain DNS for the new provider. Only worth it for a long
  outage; password resets are the urgent path.
- **Missed a day's digest:** a forced run resends it (owner approval needed:
  `gh workflow run scheduled-scans.yml --ref main -f force=true`).

### Scheduler: cron-job.org

- **Symptom:** no Scheduled Scans runs at the expected times (12:35, 13:35, 16:35,
  19:35, 20:35, 21:35 UTC on weekdays).
- **Cause is usually** the GitHub token in cron-job.org expired, or cron-job.org
  disabled the job after repeated failures. Renew the token / re-enable the job.
- **Meanwhile:** start runs by hand with `gh workflow run scheduled-scans.yml --ref main
  -f force=false -f session=auto`. GitHub's own `schedule:` trigger was removed on
  purpose (it fired hours late); don't re-add it without deciding that again.

### GitHub Actions

- **Outage:** check githubstatus.com. Nothing to do but wait; the next cron slot
  runs after GitHub recovers. The daily emails catch up inside their windows.
- **Failing run:** open the run's log; Autonomous recovery (Run 60) retries the
  allowlisted problems on its own. See `docs/AUTONOMY_OPERATOR_CARD.md`.

### Render (billing service)

- **Symptom:** checkout or the billing portal fails, `/health` is not `ok`.
- `/health` lists missing settings and whether the database is reachable. Fix the
  setting in Render; a redeploy picks it up.
- **Bad deploy:** in Render → Deploys, roll back to the previous deploy, then fix on
  `dev`.
- Stripe retries webhooks for about three days, so a short outage loses no
  upgrades; processed events are de-duplicated.

### Streamlit Community Cloud

- **App crash after a deploy** is most often a stale module (an old copy of a
  changed module still loaded). Reboot the app from the Streamlit Cloud menu.
- **Outage:** nothing to fail over to; scheduled scans and emails keep running on
  GitHub Actions.

### Anthropic (AI summaries)

- **Symptom:** Premium AI summaries time out or error.
- Summaries fail on their own (30 s timeout) without blocking the page. For a long
  outage set `AI_ENABLED=0` on Streamlit Cloud to hide AI features cleanly; set it
  back to `1` afterwards.

### Stripe

- Checkout and the portal fail with a clear message while Stripe is down. Nothing
  to do; paid plans already recorded in the database keep working.

## Neon: restoring the live database

The live app reads and writes the Neon branch **`live`** (the project's default
branch). Its parent branch, `stale-production-unused`, holds old data.

**Never use "Reset from parent" on `live`.** That replaces live data with the old
parent data.

Restore window: **1 day** on the current plan. Anything older can't be restored,
so act the same day.

### Before restoring

1. Pause the scan job in cron-job.org so no scheduled run writes mid-restore.
   App users may see a brief error while the branch restores; that's expected.
2. Pick the restore time: the last moment the data was good (UTC).

### Restore (console)

Neon console → the project → Branches → `live` → **Restore** → choose the time →
keep "Preserve current state as a backup branch" on → Restore. The branch keeps its
name and connection string, so no secrets change.

### Restore (CLI)

```bash
npx -y neon@latest branches restore live ^self@2026-09-29T14:00:00Z --preserve-under-name live_before_restore --project-id <project id>
```

Replace the timestamp with the restore time and `<project id>` with the id from the
Neon console. Check `npx -y neon@latest branches restore --help` first if unsure;
`--preserve-under-name` keeps the pre-restore state as a branch so the restore can
be undone.

### After restoring

1. Check the app signs in and the Admin → System health card.
2. Re-enable the cron-job.org job and run one scan by hand.
3. Delete the backup branch (`live_before_restore`) once you're sure; Neon counts
   its storage.
4. Everything written after the restore time is gone: scan runs, research
   observations, sign-ups, settings. Note the gap on the tracker; it is not
   re-created. Check recent Stripe upgrades against user plans (Stripe dashboard →
   Events) and re-send any webhook for an upgrade made in the lost window.

### Recommended once (owner, Neon console)

- Mark `live` as a **protected branch**, so it can't be deleted or reset by mistake.
- Consider a longer restore window before scaling paid users (P1-44).
