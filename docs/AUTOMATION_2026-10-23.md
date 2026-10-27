# Automation and emails (2026-10-23)

Eight additions. Everything runs on its own after the promote; nothing new to set on
Render or GitHub. Each piece has an off switch (set the variable to `0` on hsf-api-v1).

| What | Where it runs | When | Off switch |
|---|---|---|---|
| Live check after every promote | GitHub `live-acceptance.yml` (push to main) waits for both Render services to serve the commit (`scripts/wait_for_deploy.py`, `/readyz` and `/api/healthz` now report `build`), then runs the API + browser + accessibility checks | each promote | (workflow) |
| Pass/fail message for that check | API thread `api/ops_jobs.py` reads finished runs from GitHub every 10 min, messages admins' linked Telegram/Discord (`services/ops_notify.py`); failures also email. The daily 15:15 UTC run only reports failures | within ~10 min of the run ending | `HSF_LIVE_CHECK_NOTIFY` |
| Nightly ops note | API thread, Tue-Sat 00:05-03:00 UTC: health status/score, top incidents (minus `HSF_OPS_NOTE_IGNORE`, default `RETIREMENT_POLICY_EFFECT`), flagged scan slots, API memory peak, last live check | after the 23:45 UTC health run | `HSF_NIGHTLY_OPS_NOTE` |
| Weekly recap email | API thread, Saturday from 14:00 UTC (`services/weekly_recap.py`). Last week's top-5-per-day picks vs SPY next day, watchlist movers; Pro+ also alerts fired and next week's earnings; Starter gets a teaser | weekly | `HSF_WEEKLY_RECAP_ENABLED`; per user: "Weekly recap" in Account |
| Payment failed / plan ended emails | `billing_service/engine.py` after each Stripe snapshot (`services/billing_emails.py`) | when Stripe reports it | `HSF_BILLING_EMAILS_ENABLED` |
| "Update your card" banner | `/v1/me` `billing_attention` + `web/src/components/BillingBanner.tsx` | while past due | — |
| Free Pro ended email | API thread daily from 15:00 UTC (`run_trial_ended`) | within 3 days of the end | `HSF_LIFECYCLE_EMAILS_ENABLED` |
| First-week tips + 14-day come-back | API thread daily (`services/onboarding_emails.py`); last-active comes from `db/user_activity.py` (written at most hourly per account) | days 2-6 / 5-9 / 14-30 away | `HSF_ONBOARDING_EMAILS_ENABLED`; per user: "Tips and reminders" |
| Monday business email | API thread, Monday from 12:00 UTC (`services/business_report.py`), chat + email to admins | weekly | `HSF_BUSINESS_EMAIL_ENABLED` |
| Dependabot auto-merge | `dependabot-automerge.yml` + `scripts/dependabot_merge_check.py`; Dependabot now targets dev | when all checks on the PR are green | `DEPENDABOT_HOLD` repo variable (default `typescript,eslint`) |
| What's new card | `web/src/content/whatsNew.ts` (add an entry for user-facing changes), shown once per browser | after the deploy | — |

`HSF_OPS_JOBS_ENABLED=0` stops the whole API jobs thread; `HSF_OPS_NOTIFY_ENABLED=0`
silences every ops message. Ops email goes to `HSF_OPS_ALERT_EMAIL` or the admin accounts.

Notes
- The come-back email needs 14 days of last-active data, so the first ones go out
  about two weeks after the deploy.
- PR titles were too technical for the What's new card, so it reads a short hand-written
  list instead; each feature PR adds a line.
- Old Dependabot PRs that target main stay as they are; new ones open against dev.
- Auto-merge is a `workflow_run` workflow, so it only runs once it is on main.
