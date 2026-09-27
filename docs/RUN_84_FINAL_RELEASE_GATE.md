# Run 84 — Final Release-Gate Recheck

## Final Verdict

# CONDITIONAL GO

Two conditions apply:

- **Pro:** an invite-only paid beta with **Pro** can go ahead as soon as a live Stripe
  test-mode round trip and three short manual checks pass (§Launch Conditions).
- **Premium:** hold it until two claim-accuracy fixes to its AI features are made (P1,
  below), or sell it explicitly as a beta once they are.

No P0 remains. Every Run 82 blocker and Run 83B defect is fixed. Where they touch a live
service, they were verified on it.

## Release Candidate

| Item | Value |
|---|---|
| main = dev | `7011f5469000668b60f3cbf87b1951396bd86b57` (Run 83C) |
| Recheck time | 2026-09-27 04:28 UTC (Saturday; market closed) |
| CI on main | Smoke Checks: all five jobs green, including `dependency-audit` (first green run) |
| Autonomy certification | **AUTONOMOUS_RESEARCH_MODE_V1_CERTIFIED** at `7011f54` (run 36293990647): 24/24 mandatory gates incl. Gate U (frozen scanner); INFO_1 resolved (cron-job.org timezone confirmed UTC) |
| Billing service (Render) | Running the Run 83 code (verified below) |
| App (Streamlit Cloud) | Running the post-83B code; no startup error |

Everything was verified read-only, with one exception: a single AI-summary request, made
to prove the Premium AI configuration. Live checks used the owner's already-signed-in
browser session (Basic plan with admin tools and a populated watchlist). No other
account, credential or payment was used.

## Blocker Recheck

| Finding | Status | Live evidence |
|---|---|---|
| **B1** Billing portal authentication | **FIXED, LIVE** | Billing service: `/health` 200 (no missing env, DB reachable). `/debug/status` 404. Unsigned `POST /create-portal-session` → **401** "Please sign in again to manage billing." Unsigned `POST /create-checkout-session` → **401**. Forged `X-HSF-Auth` → **401** "could not be verified" (the live token check reaches the database). Probes used a non-existent `@invalid.example` address, so no customer could be affected on any code version. |
| **B2** Account-state isolation | **FIXED** (code + CI) | 10 isolation tests and 11 Scanner cross-account tests green in CI. Live A → logout → B not exercised (needs two accounts): **condition C3**. |
| **B3** Scanner shows My Watchlists | **FIXED, LIVE** | Owner account, populated watchlist, Today → Scanner: `Scanner` → `Results` → `HSF Opportunities` rendered immediately with "Latest full-market scan · Fri Sep 25, 3:40 PM ET (32 h ago) · 100 ranked setups". `My Watchlists` rendered afterwards (heading 12 vs 5), after its quote fetch. The Premium account itself was not available (tier is incidental; all tiers are covered in CI): **condition C3**. |
| Stripe restore token | **SINGLE-USE** (code + CI) | 13 tests in CI. Live Stripe round trip not exercised: **condition C1**. |
| Dependency CVEs (P1-21) | **DONE** | tornado 6.5.8, cryptography 50.0.0, soupsieve 2.9, GitPython 3.1.60; `pip-audit`: no known vulnerabilities; CI `dependency-audit` green |
| Startup / stale-module recovery | **HEALTHY** | No startup error after two redeploys; both import blocks self-heal (bounded); regression tests in CI |

**Also verified live:**
- A new session lands on **Today** once (snapshot, top setups, "new since your last
  visit", watchlist, recap).
- The Scanner shows the canonical `cron`/`US_MARKET` scan, with its age and row count.

## Premium AI Check (Run 82 L3)

- **Configured:** yes. One "✨ Generate AI summary" request returned a Claude summary with
  no configuration or authentication error, so the Anthropic key and model are valid in
  production.

While checking this, two claim-accuracy problems turned up (new findings, not fixed
here):

| ID | Severity | Problem | Evidence | Needed before selling Premium? |
|---|---|---|---|---|
| **R84-1** | P1 | The **"AI Notes"** panel (sold as "AI setup notes") is **not AI**. `ui/ai_notes.generate_ai_note` fills a fixed template ("MXL shows a breakout score of 52.3. Gap is 2.10%…"). | `ui/ai_notes.py:84-97` (`_breakout_note` / `_quote_note`); live panel text | Yes: rename it (e.g. "Setup notes"), or generate it with Claude |
| **R84-2** | P1 | **AI Scan Summary** conflicts with the product's own rules. It ranks by **BreakoutScore**, never the HSF Score (the prompt columns omit it, against the Run 79 hierarchy). Its output uses trading-action language: "confirm with intraday support holds **before entry**", "adequate for **position size**", "signals **institutional accumulation**". This is despite the prompt's "do NOT give financial advice", and no disclaimer is shown on screen. | `ui/ai_summary.py:12-26` (`_SUMMARY_COLUMNS`, `_SYSTEM_PROMPT`); live output | Yes: include HSF Score and rank by it; tighten the prompt to descriptive language (no entry, sizing or institutional inferences); show the standard "research tool, not financial advice" caption under AI output (summary and chat) |

These don't affect Free or Pro. Premium's headline value is these AI features, so they
should be accurate before anyone pays $39 for them.

## Launch Conditions

These have to be done by a person, with accounts I can't use:

| # | Condition | Why |
|---|---|---|
| **C1** | **Stripe test-mode round trip on the live app:** sign up → upgrade to Pro → webhook → Pro visible → **Manage subscription** opens the portal → switch to Premium → cancel → back to Free. | Run 83 changed how the app authenticates to the billing service. Code and CI cover it, but the live checkout and portal path with the new token hasn't been exercised. This is the one check that must pass before taking real money. |
| **C2** | **Email:** sign-up verification, password reset, one Pro alert email. | Delivery can't be verified from the repository. |
| **C3** | **Live account and phone checks:** a Premium account with a watchlist → Scanner shows results first; sign in as A → log out → sign in as B in the same tab → B's Market Brief watchlist is B's own; a quick signed-in walk through Today / Scanner / Stock / My Stocks at phone width (~390 px). | Closes B2 live, the Premium side of B3 live, and P2-13. |
| **C4** | **Premium only:** fix R84-1 and R84-2, or don't offer Premium in the first cohort. | Claim accuracy of a paid tier. |

**Housekeeping (not blocking):**
- Confirm the Streamlit Cloud app's Python version in its settings (the repo has no
  `runtime.txt`; CI uses 3.13).
- Confirm the 21:35 UTC slot is in the cron-job.org job; the screenshot showed only the
  next five runs.
- Watch the `MATURATION_CAP_BINDING` incident reported by the certification snapshot
  (system DEGRADED, score 90, automatic recovery enabled).

## Remaining Open Items (not launch-blocking)

| ID | Severity | Item |
|---|---|---|
| F8 (Run 82) | P2 | "Most popular" badge on Pro with no customers (`ui/pricing.py:183`, `pages/billing.py:246`) |
| F9 (Run 82) | P2 | The landing page doesn't say what the HSF Score represents (Methodology does) |
| F12 (Run 82) | P2 | FastAPI `on_event` deprecation in the billing service |
| P2-13 | P2 | Signed-in phone QA, covered by C3 |
| P2-19 / Run 85 | Blocked | Run your own scans off the page request (frozen core); evidence page (waits on Run 56) |

## Launch Scope Recommendation

- **Scope:** invite-only paid beta, starting with people you can contact directly.
- **Pro ($19): yes, once C1–C3 pass.** Its value (email alerts, 5 alerts, CSV and the
  interactive table, Nasdaq and combined scans, earnings, history) is implemented,
  enforced server-side, and covered by CI.
- **Premium ($39): only after C4.** Then label the AI features "beta".
- **Free: yes.** Nothing blocks it.
- **Scale:** no load testing was done (Streamlit Cloud, Neon connections, Render). Grow
  invites gradually while watching the system-health workflow and the Render logs; don't
  commit to a user count.

## Decision Rationale

- **Evidence behind the conditional GO:**
  - every verified launch blocker from Run 82 (B1, B2) and the post-83 defect (B3) is
    fixed;
  - where they touch a live service, the fix was proven there (the billing service
    rejects anonymous, unsigned and forged requests; Scanner renders canonical results
    first);
  - dependencies are clean;
  - the research core is re-certified.
- **What remains:**
  - a live payment round trip, which should be exercised by a person before the first
    real charge;
  - two accuracy fixes to Premium's AI features.
- **Why not unconditional GO:** C1 is the path that changed most recently and touches
  money, and it hasn't run live.
- **Why not NO-GO:** no open issue makes a controlled Pro beta unsafe or misleading.

**Next step:** complete C1–C3 by hand. If they pass, launch the Pro invite-only beta. For
Premium, a small **Run 85** limited to R84-1 and R84-2, then offer Premium as beta.
