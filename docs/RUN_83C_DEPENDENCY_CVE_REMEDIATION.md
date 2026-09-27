# Run 83C — Dependency CVE Remediation

## Executive Summary

**P1-21 is DONE.** Each of the four flagged packages was checked individually:

- **tornado:** already patched by Run 83 (6.5.8). The audit reports nothing, so it was not
  changed again.
- **cryptography, soupsieve and GitPython:** moved to the smallest releases that fix every
  flagged advisory. That is exactly **three pin changes** in `requirements.lock`. No
  transitive package moved, and the full resolved environment differs from before only in
  those three.
- **Audit result:** `pip-audit` reports **no known vulnerabilities**, both for the lock
  (strict, as in CI) and for the installed production environment. It reported 31 before.
- **Regression:** the full suite, the Run 83 security tests, the Run 83B Scanner and
  boot-recovery tests, the frozen-core/research tests, the app boot and every page all
  pass.

## Baseline

| Item | Value |
|---|---|
| Branch / HEAD before | `dev` @ `7f739d4` (= main) |
| Python | CI 3.13 (all workflows); local production-parity venv 3.12.6 |
| Streamlit | 1.54.0 |
| Dependency sources | `requirements.txt` → `-c requirements.lock` + `requirements-core/ml/extended.txt`; `requirements-dev.txt`; `requirements-billing-test.txt` (FastAPI/httpx only); `billing_service/requirements.txt` (none of the four); `pyproject.toml` (`beautifulsoup4`). No Dockerfile. The CI `dependency-audit` job runs `pip-audit -r requirements.lock --strict`. |

| Package | Locked | How it's installed | Used by | Production reachable? |
|---|---|---|---|---|
| tornado | 6.5.8 | transitive (streamlit `>=6.0.3,<7,!=6.5.0`) | Streamlit's web server | Yes |
| cryptography | 49.0.0 | transitive (streamlit-authenticator `>=42.0.5`, streamlit-cookies-manager); also imported directly by `db/secret_box.py` | Login cookie encryption (PBKDF2 + Fernet), encrypted Alpaca paper keys (Fernet) | Yes, but PKCS#7 is not used (see below) |
| soupsieve | 2.8.4 | transitive (beautifulsoup4 `>=1.6.1`; bs4 is declared in `requirements-core.txt`) | CSS selectors in bs4 | No first-party code imports bs4 or calls `.select()`; only third-party code could reach it |
| GitPython (+ gitdb 4.0.12, smmap 5.0.3) | 3.1.51 | transitive (streamlit `>=3.0.7,<4,!=3.1.19`) | Streamlit's git-repo detection | No first-party `import git`; no user input reaches it |

## Advisory Inventory

Source: `pip-audit 2.10.1 -r requirements.lock --no-deps --disable-pip --desc`
(PyPI/OSV). Locally, the lock can't be trial-installed under Python 3.12, and every line
is an exact pin, so the pins were audited directly. CI audits the same file on 3.13.

| Package | Advisory (aliases) | Affected | Fix | Nature | HSF exposure | Status before |
|---|---|---|---|---|---|---|
| tornado 6.5.8 | none | — | — | — | — | **PATCHED ALREADY** (Run 83: PYSEC-2026-3928, GHSA-wwv5-g3v4-889x, GHSA-8423-8fgw-73vq, fixed in 6.5.8) |
| cryptography 49.0.0 | PYSEC-2026-3552 (GHSA-g6cj-pr64-35w5, CVE-2026-69247) | < 50.0.0 | 50.0.0 | `pkcs7_decrypt_der/pem/smime` report RecipientInfo decryption outcomes distinguishably (oracle-style) | PKCS#7 decryption is never called by HSF or its libraries (only Fernet/PBKDF2) → not reachable | REQUIRES REMEDIATION (defense in depth) |
| soupsieve 2.8.4 | CVE-2026-85999 (GHSA-j934-xhv5-fg8f); CVE-2026-86000 (GHSA-gjv8-xp57-g29c) | < 2.9.0 | 2.9.0 | Regex DoS when compiling attacker-supplied CSS selectors | Selectors are never user-supplied or first-party → transitive but unreachable | REQUIRES REMEDIATION (defense in depth) |
| GitPython 3.1.51 | 21 advisories: PYSEC-2026-3984, -3982, -3981, -3951, -3783, -3785, -3786, -3787, -3788, -3784, -3837, -3840, -3838, -3841, -3843, -3949, -3953, -3952, -3950, -3948, CVE-2026-73624 (28 audit rows, some duplicated per source) | < 3.1.53 … < 3.1.60 | **3.1.60** (highest) | Option/argument injection, config injection, path traversal and ReDoS in clone/config/submodule/blame/diff APIs | Only Streamlit's local repo detection uses it; no attacker-controlled repos, URLs or options → transitive but unreachable | REQUIRES REMEDIATION (defense in depth) |

None of the three was exploitable through HSF as deployed. They are patched anyway
because the fixes are available, small and compatible.

## Tornado

- 6.5.8 (Run 83) resolves all three advisories flagged in Run 82.
- The current audit reports **no** tornado findings.
- 6.5.9 and 6.5.10 exist, but no applicable advisory requires them.
- **Status: PATCHED. No change in this run.**

## cryptography

- **Change:** 49.0.0 → **50.0.0**, the minimum patched release.
- **Requirements of 50.0.0:** `cffi>=2.0.0` (lock has 2.1.0), and `typing-extensions` only
  for Python < 3.11.
- **Other constraints:** `curl_cffi`'s `cryptography<47` bound applies only to its dev/test
  extras, which aren't installed. No other installed package caps cryptography.
- **APIs in use:** `cryptography.fernet.Fernet` (`db/secret_box.py`,
  streamlit-cookies-manager, streamlit-authenticator), `hazmat.primitives.hashes` and
  `kdf.pbkdf2.PBKDF2HMAC`. These are long-stable APIs; no code changed.
- **Verified on 50.0.0:**
  - `secret_box.encrypt_secret`/`decrypt_secret` round-trip;
  - the cookie manager's PBKDF2-SHA256 + Fernet scheme round-trip;
  - `EncryptedCookieManager` imports;
  - `tests/test_paper_trading.py` (encrypted keys), `test_p2_backlog` and
    `test_run62_trust_layer` (cookie paths) pass.
- **Run 83 tokens:** they use `hashlib`/`secrets`, not cryptography.

## soupsieve

- **Change:** 2.8.4 → **2.9** (PyPI's version string for 2.9.0), the minimum patched
  release; needs Python ≥ 3.10.
- **How it's installed:** via `beautifulsoup4 4.15.0` (`soupsieve>=1.6.1`), which is
  declared in `requirements-core.txt`.
- **Usage:** no first-party module imports bs4, `BeautifulSoup` or calls `.select()`.
  The universe and web data paths use the Alpaca API and pandas, not bs4.
- **Verified:** bs4 `select('p.a')` works on 2.9.

## GitPython

- **Change:** 3.1.51 → **3.1.60**, the highest fix version among its advisories.
- **Requirements:** `gitdb<5,>=4.0.1`, so gitdb stays 4.0.12 and smmap stays 5.0.3. Neither
  has a finding.
- **Usage:** required by Streamlit only. HSF has no `import git` in runtime, CI or research
  tooling.
- **Verified:** `git.Repo(...).head.commit` works (Streamlit's usage pattern).

## Dependency Diff

`requirements.lock`, three lines:

```
-cryptography==49.0.0
+cryptography==50.0.0
-GitPython==3.1.51
+GitPython==3.1.60
-soupsieve==2.8.4
+soupsieve==2.9
```

- **Resolved environment diff** (`pip freeze`, production-parity venv before vs after):
  exactly the same three packages. **No transitive changes.**
- **Streamlit:** still 1.54.0, tornado 6.5.8, pyarrow 23.0.1, pandas 2.3.3.
- **How the lock was edited:** the lock is normally regenerated by the Freeze Dependency
  Lock workflow. These three pins were edited by hand, following the Run 83 tornado
  precedent, to keep the change minimal, then verified by a clean install.

## Security Scan Before/After

| Package | Before | After |
|---|---|---|
| tornado | 0 findings (6.5.8) | 0 findings |
| cryptography | 1 finding (2 rows) | 0 |
| soupsieve | 2 | 0 |
| GitPython | 21 advisories (28 rows) | 0 |
| **Total** | **31 findings in 3 packages** | **"No known vulnerabilities found"** |

- `pip-audit -r requirements.lock --strict` (the CI rule): **rc 0**.
- `pip-audit` on the installed production-parity environment: "No known vulnerabilities
  found".

## Compatibility Verification

- **Clean install from scratch:** a new venv from `requirements.txt` +
  `requirements-dev.txt` with the new lock. Install rc 0; `pip check`: "No broken
  requirements found".
- **No package moved into another vulnerable range:** confirmed by the audit of the
  installed environment.
- **Streamlit startup:** `scripts/streamlit_smoke.py` rc 0. `deployment_doctor`: 22 [OK].
- **CI import smoke:** `config`, `scheduler.cron_runner`, `_load_universe('SP500')`.
- **Scheduler and research entry points import:** `scripts.mature_observations`,
  `autonomous_recovery`, `autonomy_certification`, `forward_evidence_readiness`,
  `analytics.btc_outcome_logger`, `scan.engine`, `scan.pre_post`.
- **Page smoke** (every page script run headlessly signed out, no exceptions): app
  (landing/auth), Today, Market Brief, Stock Intelligence, My Stocks, Alerts, Billing,
  Methodology, Settings, Day Trader.

## Test Results

Production-parity venv with the new lock (Python 3.12.6, Streamlit 1.54.0):

| Suite | Collected | Passed | Failed | Skipped | xfail | Warnings | Duration |
|---|---:|---:|---:|---:|---:|---|---:|
| Full, `-X dev -W always`, outbound network blocked | 1916 | 1878 | 0 | 38 (all "fastapi not installed"; covered by the billing job) | 0 | 1 (Streamlit AppTest temp dir, third-party) | 171 s |
| Lightweight CI env | 1916 | 1788 | 0 | 128 | 0 | 0 | 17.9 s |
| `unittest discover -s tests` | 1873 run | OK | 0 | 38 | — | — | 107 s |
| Billing-contract job (CI command) | 111 | 110 | 0 | 1 | 0 | 81 (FastAPI/Starlette deprecations) | 2.0 s |
| Named security/startup/research set | 206 | 206 | 0 | 0 | 0 | — | 48.6 s (159 subtests) |
| `ruff check .` | — | All checks passed | — | — | — | — | — |

The named set:
- Run 83 security: `test_run83_restore_token`, `test_run83_account_isolation`;
- Run 83B: `test_run83b_scanner_state`, `test_run83b_boot_recovery`;
- startup: `test_boot_stale_module`;
- cryptography users: `test_paper_trading`, `test_p2_backlog`, `test_run62_trust_layer`;
- frozen core and research: `test_autonomy_certification` (incl. Gate U),
  `test_maturation_e2e`, `test_observation_capture`, `test_signal_evidence`.

## CI Verification

The GitHub Actions result for this commit is recorded in the final message of this run.
No CI change was needed or made; the `dependency-audit` job is expected to pass for the
first time since it was added.

## Run 83 Security Regression

All green:
- `test_run83_billing_auth` (12, billing job): anonymous, forged, replayed and
  cross-account requests are refused.
- `test_run83_restore_token` (13): single use, replay rejected.
- `test_run83_account_isolation` (10).

## Run 83B Regression

All green:
- `test_run83b_scanner_state` (11): canonical results for every tier, populated and empty
  watchlists, slow and failing watchlists, cross-account.
- `test_run83b_boot_recovery` (4) and `test_boot_stale_module` (2): stale-module recovery
  is bounded, with no rerun loop.

## Frozen-Core Verification

- **Code:** the only file changed besides this report is `requirements.lock` (three pins).
  No change to scoring, ranking, `run_breakout_scan`, models, scheduling, Gate U, research
  capture, maturation, cohorts, Autonomous Research Mode or Run 56/58/61 behaviour.
- **Research tests:** the certification and research suites pass.
- **Re-certification:** the lock also feeds the scheduled workflows, so re-run autonomy
  certification once this is on main. This is the standing rule for lock changes, and
  covers Run 83's tornado pin as well.

## P1-21 Status

**DONE**

| Package | Before | After | Advisory | Status |
|---|---|---|---|---|
| tornado | 6.5.7 (Run 82) → 6.5.8 (Run 83) | 6.5.8 (unchanged) | PYSEC-2026-3928, GHSA-wwv5-g3v4-889x, GHSA-8423-8fgw-73vq | **PATCHED** |
| cryptography | 49.0.0 | 50.0.0 | PYSEC-2026-3552 / CVE-2026-69247 | **PATCHED** |
| soupsieve | 2.8.4 | 2.9 | CVE-2026-85999, CVE-2026-86000 | **PATCHED** |
| GitPython | 3.1.51 | 3.1.60 | 21 advisories (PYSEC-2026-3783…3984, CVE-2026-73624) | **PATCHED** |

## Remaining Risk

- **No known dependency vulnerability remains** in the locked or installed set as of this
  audit. New advisories will surface through the existing CI `dependency-audit` job.
- **Hand-edited lock:** the three pins were changed by hand. The next run of the Freeze
  Dependency Lock workflow should keep them at or above these versions; check the diff
  when it runs.
- **Python version gap:** production parity was verified on Python 3.12; CI covers 3.13.
  Confirm the Streamlit Cloud app's Python version in its settings (no `runtime.txt`), a
  standing item for the Run 84 checklist.

Next: **Run 84, the final release-gate recheck.**
