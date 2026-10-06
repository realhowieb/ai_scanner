// Post-deploy smoke check for a Web v2 deployment. No browser and no credentials needed;
// with a test account (HSF_TEST_EMAIL + HSF_TEST_PASSWORD_FILE) it also checks sign-in,
// the cookie flags and signed-in timings. Never prints credentials, tokens or bodies.
//   BASE_URL=https://<beta> node scripts/beta-smoke.mjs
import { readFileSync } from "node:fs";

const BASE = (process.env.BASE_URL || "").replace(/\/+$/, "");
if (!BASE) throw new Error("Set BASE_URL");
const local = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(BASE);
const results = [];
const check = (name, ok, evidence = "") => {
  results.push(ok);
  console.log(`[${ok === null ? "BLOCKED" : ok ? "PASS" : "FAIL"}] ${name}${evidence ? ` — ${evidence}` : ""}`);
};
async function timed(path, init = {}) {
  const t = performance.now();
  const res = await fetch(BASE + path, { redirect: "manual", ...init });
  return { res, ms: Math.round(performance.now() - t) };
}

check("served over HTTPS (required for Secure __Host- cookies)", BASE.startsWith("https://") || local, local ? "localhost: http allowed for testing" : BASE);
const cold = await timed("/api/healthz");
const warm = [];
for (let i = 0; i < 3; i++) warm.push((await timed("/api/healthz")).ms);
check("health check /api/healthz", cold.res.status === 200, `first ${cold.ms} ms, then ${warm.join(" / ")} ms`);

const loginPage = await timed("/login");
const h = loginPage.res.headers;
check("sign-in page", loginPage.res.status === 200, `${loginPage.ms} ms`);
const want = { "content-security-policy": /connect-src 'self'/, "x-frame-options": /DENY/, "x-content-type-options": /nosniff/,
  "referrer-policy": /same-origin/, "strict-transport-security": /max-age=/ };
const missing = Object.entries(want).filter(([k, re]) => !re.test(h.get(k) || "")).map(([k]) => k);
check("security headers", missing.length === 0, missing.length ? `missing: ${missing.join(", ")}` : "CSP, frame, nosniff, referrer, HSTS");
check("no framework banner", !h.get("x-powered-by"));

const guard = await timed("/today");
check("protected page without a session goes to sign-in", [307, 308].includes(guard.res.status) && /\/login/.test(guard.res.headers.get("location") || ""), `${guard.res.status}`);
const me = await timed("/api/hsf/v1/me");
const meBody = await me.res.json().catch(() => ({}));
check("BFF without a session: 401 session_expired, not cached", me.res.status === 401 && meBody.code === "session_expired" && /no-store/.test(me.res.headers.get("cache-control") || ""), `${me.ms} ms`);
const auth = await timed("/api/hsf/v1/auth/refresh");
check("auth routes are not proxied", auth.res.status === 404);
const xsite = await timed("/api/auth/login", { method: "POST", headers: { origin: "https://evil.example", "content-type": "application/json" }, body: "{}" });
check("cross-site sign-in refused", xsite.res.status === 403);

const email = process.env.HSF_TEST_EMAIL;
if (!email || !process.env.HSF_TEST_PASSWORD_FILE) {
  check("signed-in checks (cookie flags, /v1/me, timings)", null, "set HSF_TEST_EMAIL and HSF_TEST_PASSWORD_FILE for a dedicated test account");
} else {
  const origin = new URL(BASE).origin;
  const password = readFileSync(process.env.HSF_TEST_PASSWORD_FILE, "utf8").trim();
  const li = await timed("/api/auth/login", { method: "POST", headers: { origin, "content-type": "application/json" }, body: JSON.stringify({ email, password }) });
  const cookies = li.res.headers.getSetCookie();
  const flags = cookies.length === 2 && cookies.every((c) => /HttpOnly/.test(c) && /SameSite=Lax/.test(c) && (local || (/Secure/.test(c) && /^__Host-/.test(c))));
  check("sign-in sets two HttpOnly cookies (Secure __Host- on https)", li.res.status === 200 && flags, `HTTP ${li.res.status}, ${li.ms} ms`);
  if (li.res.status === 200) {
    const jar = cookies.map((c) => c.split(";")[0]).join("; ");
    for (const path of ["/api/hsf/v1/me", "/api/hsf/v1/today", "/api/hsf/v1/scans/latest?limit=25", "/api/hsf/v1/watchlists", "/api/hsf/v1/alerts", "/api/hsf/v1/alerts/types"]) {
      const a = await timed(path, { headers: { cookie: jar } });
      const b = await timed(path, { headers: { cookie: jar } });
      check(`GET ${path.replace("/api/hsf", "")}`, a.res.status === 200, `HTTP ${a.res.status}, ${a.ms} ms then ${b.ms} ms`);
    }
    await fetch(BASE + "/api/auth/logout", { method: "POST", headers: { origin, cookie: jar } });
  }
}
const failed = results.filter((r) => r === false).length;
console.log(`\n${results.filter((r) => r === true).length} passed, ${failed} failed, ${results.filter((r) => r === null).length} blocked`);
process.exitCode = failed ? 1 : 0;
