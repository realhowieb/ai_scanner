# Run 85 — Premium AI Claim Accuracy

## Executive Summary

Both Run 84 findings are **FIXED**:

- **R84-1 · template "AI Notes":** the notes are built from fixed templates, not AI. They
  are now titled **"📝 Setup notes"**, with the source line "⭐ Premium · built from this
  row's scan metrics (not AI-generated)". Premium's pricing copy now names only features
  that really use AI.
- **R84-2 · AI ranking / trading-action language:** every customer-facing Claude answer now
  goes through one responsibility boundary, `ui/ai_guardrails.py`, applied centrally in
  `ui.ai.ask_claude` / `ask_claude_chat`:
  - HSF's scoring system owns the HSF Score and ranking;
  - the AI explains and never ranks, picks, instructs (buy/sell/hold/enter/exit/sizing/
    targets) or predicts;
  - a narrow output check drops any line that still gives a trade instruction.
- **AI Scan Summary specifically:** it now receives the **HSF Score in HSF order** and
  explains it. Before, it ranked by BreakoutScore and never saw the HSF Score. On screen it
  says who ranks and that it's not investment advice.

**Premium can be released from the Run 84 hold** (subject to the Run 84 manual
conditions). Nothing changed in scoring, ranking, pricing, packaging or entitlements.

## Baseline

| Item | Value |
|---|---|
| Branch / HEAD before | `dev` @ `592e173` (= main): B4 fix + Run 84 report |
| Run 84 | CONDITIONAL GO; Premium held for R84-1 and R84-2 (`docs/RUN_84_FINAL_RELEASE_GATE.md`) |
| B4 | FIXED (`docs/RUN_84_B4_POST_LOGIN_CONTEXT.md`) |
| AI code | `ui/ai.py` (all Claude calls), `ui/ai_summary.py`, `ui/ai_chat.py`, `ui/ai_insights.py`, `ui/ai_screener.py`, `ui/ai_confidence_explain.py`, `ui/market_brief.py` (narrative, "AI take"), `ui/three_step_scanner.py` (AI setup), `ui/ai_notes.py` (template) |
| Entitlement | `can_ai_notes` → Premium (and admin); gates setup notes, AI summary and chat |
| Existing AI tests | none for claim accuracy |

## AI Claim Inventory

| Surface | Label / copy before | Actual implementation | Accurate before? | Action |
|---|---|---|---|---|
| Results "AI Notes" (4 render paths in `ui/results.py`) | "AI Notes" · "⭐ Premium feature" | **Template / rule-based** (`ui/ai_notes.generate_ai_note`: `_breakout_note` / `_quote_note`) | **No** | Renamed "📝 Setup notes" plus "(not AI-generated)" source line; fallback strings and comments updated |
| Pricing row / Premium card / benefits (`ui/pricing.py`) | "AI setup notes, scan summaries and results chat" | Summaries and chat = **real AI**; notes = template | **No** | "AI scan summaries and results chat, plus setup notes on each result" |
| Billing Premium caption (`pages/billing.py`) | "…AI research…" | No distinct "AI research" feature | **No** | "…AI scan summaries and chat, setup notes…" |
| AI Scan Summary (`ui/ai_summary.py`) | "Claude reviews your top results and highlights the strongest setups." | **Real AI** (Claude); ranked by BreakoutScore; no HSF Score in input; prompt "identify the 3-5 strongest" | **No** (AI picking winners; not in HSF order) | HSF-ordered input with HSF Score; explain-only prompt; caption: "Claude (AI) explains the top results in HSF Score order. HSF's scoring system does the ranking; this summary describes it. Research commentary, not investment advice." |
| Ticker deep-dive (`ai_summary.generate_ticker_analysis`) | — | Real AI | Prompt lacked HSF Score | Explain-only prompt incl. HSF Score + shared rules |
| Results chat (`ui/ai_chat.py`) | Example "Which is least risky?" | Real AI | Invited ranking and advice | Framing caption (AI, not investment advice); descriptive example |
| Scan diff / watchlist digest / alert line (`ui/ai_insights.py`) | "AI watchlist digest", "One-line AI rationale" | Real AI | Labels accurate; no guardrails | Shared rules + output check |
| Market Brief narrative and "AI take" (`ui/market_brief.py`) | "**AI take:**" | Real AI | Label accurate | Shared rules + output check |
| NL screener chips (`ui/ai_screener.py`) | "🧠 AI Stocks", "💎 Swing Trades" | Real AI → filter JSON; **no sector or trade-type filters exist** | **No** (can't deliver) | Replaced with expressible examples ("S&P 500 only", "Unusual Volume") |
| 3-step scanner upgrade name (`ui/three_step_scanner.py`) | "EZ 3-Step AI Scanner" | Deterministic scanner; only its optional "✨ AI setup" uses Claude | **No** | "3-step custom scanner" |
| AI Confidence explainer (admin expander) | "Model details · 5D outcome probability" / "Explain this score" | Real AI; admin only | Yes | Shared rules apply |
| Model outputs (5D outcome probability, PreBreakout) | Model details | **ML model output** | Yes (Run 79) | — |
| Brand "HSF AI Stock Scanner" / "HSFinest.AI" | Product name | Product includes ML models (PreBreakout XGBoost, 5D outcome model) and Claude features | Acceptable as a brand; the HSF Score itself is never attributed to AI | — |
| Landing, Methodology, Today, Stock Intelligence, Historical Research, Track Record, onboarding, emails | No AI claims beyond the brand | — | Yes | — |

## AI Notes

### Actual Implementation

- **How the note is built:** `ui/ai_notes.generate_ai_note(row)` is deterministic string
  formatting.
  - Breakout-scan rows get "{T} shows a breakout score of {x}. Gap is {g}%, with a bullish
    20-day trend of {t}%…".
  - Watchlist quote rows get a price snapshot.
  - Otherwise a fixed fallback.
- **No AI:** no model is called.
- **Name kept:** the function name stays `generate_ai_note` for callers.

### Previous Claim

The panel was headed "AI Notes", and Premium was sold on "AI setup notes".

### Corrected Claim

- **Headline:** "📝 Setup notes".
- **Source line:** "⭐ Premium · built from this row's scan metrics (not AI-generated)".
- **Fallbacks:** "Setup notes need a selectable ticker." (this also drops the incorrect
  "upgrade to Pro/Premium", shown to users who already have access); "Setup notes are
  unavailable…"; "Setup note unavailable for this row."
- **Pricing:** notes are now listed as "setup notes", separate from the real AI features.

### Fallback Behavior

- The notes never claim to be AI, whether or not AI is configured.
- The AI Scan Summary has **no template fallback**. On any provider failure (no key,
  disabled, quota, rate limit, timeout, empty response, error) it shows the error in an
  `st.warning`, and nothing is presented as AI output.
- Tested with a fake provider for auth error, rate limit, timeout, empty response and
  missing credentials: each returns `(None, error)`, and with no credentials no call is
  made.

## AI Summary

### Responsibility Boundary

```
HSF scoring system (headline_score / results_intelligence, unchanged)
   → HSF Score + canonical order
AI (Claude), where called
   → explains / summarises the data it is given, in that order
```

`ai_summary._build_table_text` orders rows with `headline_score.add_hsf_score_column` +
`rank_hsf_opportunities`, the same display pipeline the Scanner uses (Run 76), and
includes the `HSF Score` column. The HSF Score is not changed or recomputed differently.

### Prompt Changes

- **Summary:** "given the top rows of HSF's latest scan, already ordered by HSF Score
  (HSF's own opportunity score). Take the first 3-5 rows in that order and describe what
  the data shows … including any conflicting or weak readings … end with one short
  sentence on the main risks or uncertainties."
- **Removed:** "Identify the 3-5 strongest setups".
- **User message:** "top N scan results in HSF Score order … Explain the leading setups in
  this order."
- **Ticker deep-dive:** the same explain-only framing, including the HSF Score.

### Trading-Language Guardrails

- **Prompt (primary fix):** `ai_guardrails.RULES` is appended to the system prompt of
  **every** customer-facing Claude feature. It says:
  - HSF's system calculated the HSF Score and order; never claim to have calculated,
    ranked, picked or chosen them; keep the order;
  - separate facts from interpretation; mention conflicting evidence;
  - don't invent or infer data (e.g. institutional buying);
  - never tell the reader to buy, sell, hold, short, enter, exit, size a position or
    take/avoid a trade, and no entry/exit levels, stops or targets;
  - don't predict or promise outcomes; this is not personalized advice.
- **Output check (defense in depth):** `ai_guardrails.scrub` removes whole lines matching
  instruction or promissory patterns. That covers buy/sell as verbs, "short it/go short",
  enter/exit/entry, position sizing, take/avoid the trade, stop-loss, price targets,
  "will rise/break out/outperform", "guarantee" and "institutional
  buying/accumulation/participation".
  - It appends a note when it removes anything.
  - It never rewrites a line. A market fact that shares a line with an instruction is
    removed along with it, not edited.
  - Ordinary market wording is not flagged ("sell-off", "buyers", "support holds", "short
    interest").
- **Where it runs:** centrally in `ui.ai`, keyed on the existing `feature=` argument. The
  function signature is unchanged, so a stale cached module during a redeploy can't break
  callers (the Run 83B lesson).
- **Exempt:** only the JSON-returning `nl_screener`/`nl_scan_setup` and internal
  `admin_triage`. A test asserts no other feature opts out.

## Premium Packaging Verification

- **Run 73 architecture unchanged:** Free = Discover, Pro = Monitor & Investigate,
  Premium = Research & Workflow.
- **Prices unchanged:** $0 / $19 / $39.
- **Premium still adds** (derived from the entitlement map): 25 alerts; AI scan summaries
  and results chat, plus setup notes on each result; Early Breakout candidate research;
  full-market custom scanning workflow; Alpaca paper-trading workflow. Every item is a
  shipped capability.
- **Consistent everywhere:** public cards (live, verified), the Billing table (live,
  verified), the Billing Premium caption, and upgrade messages (derived from the same
  source).

## Tier Verification

- **The AI gate is unchanged:** `can_ai_notes` → Premium.
- **`compute_entitlements`:** Basic ✗, Pro ✗, Premium ✓, Admin ✓ (tested).
- **Other Premium gates** (Early Breakout, full universe, paper trading) are untouched.

## B1/B2/B3/B4 Regression

| Blocker | Tests | Result |
|---|---|---|
| B1 billing authorization | `test_run83_billing_auth` (billing job) | pass (110 passed in the billing job) |
| B2 account isolation | `test_run83_account_isolation` | 10 passed |
| B3 Scanner canonical-first | `test_run83b_scanner_state` | 11 passed |
| B4 first render after sign-in | `test_b4_post_login_context` (4 tiers, shared link, 6 transitions) | 3 passed |
| Startup / stale-module | `test_run83b_boot_recovery`, `test_boot_stale_module` | 6 passed |

## Dependency Audit

`pip-audit -r requirements.lock --strict`: **No known vulnerabilities found**. No
dependency changes in this run.

## Test Results

**New: `tests/test_run85_ai_claims.py`, 22 tests** (all run in the lightweight CI env and
the dependency job). They cover:
- guardrail rules;
- feature classification: every `feature=` in `ui/` is either guarded or one of the three
  exemptions;
- scrubbing the actual Run 84 live output;
- no false positives on ordinary market wording;
- instruction forms flagged;
- `ask_claude` wiring with a fake provider (rules sent, output checked, JSON untouched,
  failures return errors, no credentials means no call);
- summary input has the HSF Score in HSF order;
- prompts don't pick winners;
- template notes not labelled AI, and note text never says "AI";
- real-AI surfaces say they're AI and who ranks;
- the failure path shows the error, not template text;
- no copy claims AI ranks, calculates, picks or recommends;
- screener chips express only real filters;
- the 3-step scanner isn't called an AI scanner;
- Premium packaging;
- Premium's distinct value and prices;
- the tier gate.

**Against the pre-Run 85 files, 8 of the 11 labelling, packaging and summary tests fail.**
The 3 that pass were already correct: the no-AI-ranking-claim scan, the failure path and
the tier gate.

**Results** (production-parity venv, locked deps, outbound network blocked):

| Suite | Collected | Passed | Failed | Skipped | xfail | Warnings | Duration |
|---|---:|---:|---:|---:|---:|---|---:|
| Full, `-X dev -W always` | 1941 | 1902 | 0 | 38 (all "fastapi not installed"; covered by the billing job) | 0 | 1 (Streamlit AppTest temp dir, third-party) | 146 s |
| Lightweight CI env | 1941 | 1810 | 0 | 131 | 0 | 0 | 17.7 s |
| `unittest discover` | 1898 run | OK | 0 | 38 | — | — | 98 s |
| Billing-contract job | 111 | 110 | 0 | 1 | 0 | 81 (FastAPI/Starlette deprecations) | 1.9 s |
| Run 85 AI claims (final) | 22 | 22 | 0 | 0 | — | — | 1.1 s |
| Frozen core / research (certification incl. Gate U, maturation, observation, signal evidence) | 95 | 95 | 0 | 0 | — | — | 14.6 s |
| Hierarchy / pricing (Runs 79, 72, 73) | 27 | 27 | 0 | 0 | — | — | 1.0 s |

- **Lint** clean. **Boot smoke** rc 0.
- **Line budgets:** `app.py` 839 (under 840); `ui/results.py` 783 (under 795).
- **GitHub Actions** on `f1979fe` (run 36296407489): all five jobs green, including
  `dependency-audit`.

## Manual Verification

| Surface | Result | Basis |
|---|---|---|
| Pricing (landing Premium card) | **PASS** | Live, signed out: "AI scan summaries and results chat, plus setup notes on each result"; no "AI setup notes" |
| Billing (signed out) | **PASS** | Live: the same row, ❌ Free, ❌ Pro, ✅ Premium; no "AI setup notes" or "AI research" |
| Scanner Setup notes panel | UNVERIFIED live (PASS in source and tests) | Needs a signed-in Premium/admin session; the browser-pane session had expired |
| AI Scan Summary (caption and output) | UNVERIFIED live (PASS with a fake provider + the actual Run 84 output) | Same |
| Stock Intelligence | PASS | No AI claims on this page (inventory) |

- **Can a reasonable user still believe template content is AI-generated? NO.** The panel
  is titled "Setup notes" and says "not AI-generated"; pricing lists setup notes
  separately from the AI features.
- **Can a reasonable user believe AI itself ranks the market or tells them what to trade?
  NO.**
  - The summary caption says HSF's scoring system does the ranking and the summary is
    research commentary, not investment advice.
  - The prompts forbid ranking claims and trade instructions.
  - The output check removes instruction lines.

**Owner check:** on a Premium account, open Scanner, click "Generate AI summary" and read
the output. It should be in HSF Score order with no entry, sizing or target language. The
"📝 Setup notes" panel should say "not AI-generated".

## Frozen-Core Verification

`git diff` over `scan/`, `research/`, `analytics/`, `scheduler/`, `jobs/`, `data/`,
`db/`, `scripts/`, `ml_prebreakout.py`, `market_data.py`, `ui/headline_score.py`,
`ui/results_intelligence.py`: **no changes.**

- `ui/ai_summary.py` only calls the existing HSF Score display helpers.
- No change to scoring, ranking, `run_breakout_scan`, models, scheduling, research
  capture, Gate U, maturation, cohorts or certified Run 56/58/61 behaviour.
- No re-certification is needed (no lock or frozen-path change).

## Run 84 Finding Reconciliation

| Finding | Before | After | Evidence |
|---|---|---|---|
| **R84-1** Template "AI Notes" | Template text headed "AI Notes"; sold as "AI setup notes" | **FIXED**: "📝 Setup notes · … (not AI-generated)"; Premium copy names only the real AI features | Live Pricing/Billing; `LabellingTests`, `PackagingAndTierTests`; 8/11 of these fail on the old files |
| **R84-2** AI ranking / trading-action language | Summary picked "strongest" by BreakoutScore without the HSF Score; output had "before entry", "position size", "institutional accumulation"; no on-screen framing | **FIXED**: HSF-ordered input with HSF Score; explain-only prompts; shared rules on every AI feature; instruction lines removed; on-screen framing | `SummaryInputTests`, `ScrubTests` (the actual Run 84 output), `AskClaudeIntegrationTests`, `GuardrailRuleTests` |

**Both FIXED, so Premium is released from the Run 84 hold.** The Run 84 manual conditions
(C1–C3 plus the B4 sign-in check) still apply before taking real payments. Premium's AI
features can be offered, labelled "beta" as recommended in Run 84.
