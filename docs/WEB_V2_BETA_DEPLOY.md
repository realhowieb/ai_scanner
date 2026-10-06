# Web v2 beta: deployment, verification and rollback

A separate Render web service for the new frontend. It runs **beside** Streamlit:
Streamlit, its domain, the billing service and hsf-api routing don't change. Removing
the beta never affects them.

## 0. Order of operations

1. **Merge PR #8 into dev**, then **promote** (dev → main). Render redeploys hsf-api from
   main. Web v2 needs two endpoints that only exist after that: `DELETE /v1/scans/{id}`
   (Cancel) and `GET /v1/alerts/types` (alert forms).
   - Check: `https://hsf-api.onrender.com/openapi.json` lists both paths.
2. **hsf-api to Starter** (Settings → Instance type). On Free it sleeps after 15 minutes
   (the first request then takes about 60 s) and has 512 MB, so restarts lose queued scans.
3. **Create the beta service** (section 1) and point it at `dev`.
4. **Verify** (section 3) before inviting anyone.

## 1. Render web service (dashboard: New → Web Service → this repository)

| Setting | Value |
|---|---|
| Name | `hsf-web-beta` |
| Branch | `dev` (the beta follows dev; production Streamlit is unaffected) |
| Root directory | `web` |
| Runtime | Node. The version comes from `web/.node-version` (22); `NODE_VERSION=22` also works. |
| Build command | `npm ci && npm run build` |
| Start command | `npm start` (`next start` listens on `$PORT`) |
| Health check path | `/api/healthz` (liveness only; it doesn't call the API, so an API blip can't restart the web server) |
| Instance type | Free is fine for a handful of testers (it sleeps after 15 minutes; first load about 30–60 s). Starter for anything wider. |
| Auto-deploy | On (each push to dev) |

Environment variables (all server-side; nothing secret):

| Name | Value |
|---|---|
| `HSF_API_BASE_URL` | `https://hsf-api.onrender.com`. Server-only: the browser never sees it. https is required; the app refuses plain http except localhost. |
| `NEXT_PUBLIC_STREAMLIT_URL` | `https://hsfinestai.streamlit.app` (or the beta Streamlit app). Used for "Open classic app", "Create an account" and "Forgot password". Read at **build** time, so a change needs a redeploy. |
| `WEB_BETA_ALLOWED_EMAILS` | Comma-separated emails of invited testers, e.g. `you@example.com,tester@example.com`. Anyone else who signs in gets a polite "invite-only" message, and their new session is revoked at once. Leave unset to allow every HSF account. |
| `NODE_VERSION` | `22` (optional; matches `.node-version`) |

Not needed: `API_CORS_ORIGINS` on hsf-api (the browser only talks to this service), and
any API keys or `API_JWT_SECRET` here.

**Cookies and HTTPS:** `npm start` runs in production mode, so session cookies are
`__Host-hsf_at` and `__Host-hsf_rt`: `Secure`, `HttpOnly`, `SameSite=Lax`, `Path=/`.
Render serves `https://hsf-web-beta.onrender.com` with TLS, which `__Host-` cookies require.
A custom domain for the beta (e.g. `beta.hsfinest.ai`) is optional; add it in Render, and
nothing in the app changes.

**Invite list caveat:** the list is checked at sign-in. Removing someone takes effect at
their next sign-in. To cut access at once, reset that account's password, which signs
it out everywhere.

## 2. What the service exposes

- Pages: `/login`, `/today`, `/scanner`, `/scanner/custom`, `/stocks/{ticker}`,
  `/watchlists`, `/alerts`. Everything except `/login` needs a session; without one it
  redirects to sign-in.
- `/api/auth/login`, `/api/auth/logout`: sign-in and sign-out (same-origin only).
- `/api/hsf/v1/...`: the BFF proxy to hsf-api (not `/v1/auth/*`).
- `/api/healthz`: liveness.
- `robots: noindex`; security headers (CSP with `connect-src 'self'`, HSTS, frame DENY,
  nosniff, same-origin referrer).

## 3. Verify (in this order)

1. **Smoke** (no account needed):
   `BASE_URL=https://hsf-web-beta.onrender.com node web/scripts/beta-smoke.mjs`.
   It checks HTTPS, health, security headers, the sign-in redirect and the BFF refusals.
   With `HSF_TEST_EMAIL` and `HSF_TEST_PASSWORD_FILE` it also checks cookie flags and
   signed-in timings for each screen's API call.
2. **API acceptance** (dedicated test accounts only):
   `API_BASE_URL=https://hsf-api.onrender.com python scripts/api_acceptance.py --allow-writes --report acceptance.json`
   with the `ACC_FREE`, `ACC_PRO`, `ACC_PRO2`, `ACC_PREMIUM`, `ACC_ADMIN` `_EMAIL` / `_PASSWORD` variables.
3. **Browser journey** through the beta:
   `BASE_URL=https://hsf-web-beta.onrender.com HSF_TEST_EMAIL=… HSF_TEST_PASSWORD_FILE=… HSF_TEST_EMAIL_2=… node web/scripts/journey.mjs`.
   - It covers sign-in, reload, Scanner → save to a new watchlist, the stock page, a
     non-firing price alert, a duplicate refused, a note, the alert turned off, sign out
     and back in, a second account seeing nothing, phone layout, and confirmed cleanup.
   - Everything it creates is named `zz-e2e-<run>`; the alert is "price above $999,999";
     it deletes only its own records.
   - The test accounts must be on `WEB_BETA_ALLOWED_EMAILS` if that is set.
4. By hand, once: a small custom scan (One ticker) completes; S&P 500 → Cancel → start
   another.

## 4. Rollback

| Problem | Do this | Effect on customers |
|---|---|---|
| Bad Web v2 deploy | Render → hsf-web-beta → Deploys → pick the last good deploy → **Rollback**. Or push a revert to dev. | None outside the beta |
| Take the beta offline | Render → hsf-web-beta → Settings → **Suspend** (or delete the service) | None; Streamlit is untouched |
| Stop new testers | Empty or edit `WEB_BETA_ALLOWED_EMAILS`, then save (Render redeploys) | Beta only |
| Bad hsf-api deploy after promoting | Render → hsf-api → Deploys → **Rollback** to the previous deploy. Then revert the commit on dev and promote again so main matches. | The API serves the mobile/Web v2 clients only; Streamlit doesn't use it |

Nothing in the beta writes to scanner, research or billing data. Watchlists and alerts
are the same records the classic app uses, so a tester's changes show up there too.
