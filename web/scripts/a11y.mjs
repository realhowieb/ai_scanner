// Accessibility check of the main signed-in pages (desktop and phone) with axe-core.
// Fails when any critical or serious WCAG A/AA rule is broken; lists the rest.
//   BASE_URL=http://localhost:3100 HSF_TEST_EMAIL=… HSF_TEST_PASSWORD_FILE=… node scripts/a11y.mjs
import { existsSync, readFileSync } from "node:fs";
import { chromium } from "playwright-core";

import { BLOCKING, describe, runAxe } from "./axe.mjs";

const BASE = process.env.BASE_URL || "http://localhost:3100";
const email = process.env.HSF_TEST_EMAIL;
const password = readFileSync(process.env.HSF_TEST_PASSWORD_FILE, "utf8").trim();
const LOCAL_CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const executablePath = process.env.CHROMIUM_PATH || (existsSync(LOCAL_CHROMIUM) ? LOCAL_CHROMIUM : undefined);
const PAGES = [
  ["today", "/today", (p) => p.getByRole("heading", { name: "Top setups" }).waitFor()],
  ["scanner", "/scanner", (p) => p.locator(".tk:visible").first().waitFor()],
  ["stock", "/stocks/AAPL", (p) => p.getByRole("heading", { name: "HSF Score" }).waitFor()],
  ["watchlists", "/watchlists", (p) => p.getByRole("heading", { level: 1 }).first().waitFor()],
  ["alerts", "/alerts", (p) => p.getByRole("tab", { name: "Rules" }).waitFor()],
  ["day-trader", "/day-trader", (p) => p.getByRole("heading", { level: 1 }).first().waitFor()],
  ["brief", "/brief", (p) => p.getByRole("heading", { name: "Market Pulse" }).waitFor()],
  ["track-record", "/track-record", (p) => p.getByRole("heading", { level: 1 }).first().waitFor()],
  ["account", "/account", (p) => p.getByRole("heading", { level: 1 }).first().waitFor()],
];

const browser = await chromium.launch({ executablePath });
let blocking = 0;
for (const [size, viewport] of [["desktop", { width: 1440, height: 1000 }], ["phone", { width: 390, height: 844 }]]) {
  const ctx = await browser.newContext({ viewport });
  const page = await ctx.newPage();
  await page.goto(`${BASE}/today`);
  await page.waitForURL(/\/login/);
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(/\/today/, { timeout: 90_000 });
  for (const [name, path, ready] of PAGES) {
    await page.goto(`${BASE}${path}`);
    await ready(page);
    await page.waitForTimeout(800); // let late cards render
    const found = await runAxe(page);
    const bad = found.filter((v) => BLOCKING.has(v.impact));
    blocking += bad.length;
    console.log(`[${bad.length ? "FAIL" : "PASS"}] ${size} ${name}: ${found.length ? `${found.length} rule(s)` : "no violations"}`);
    for (const line of describe(found)) console.log(`    ${line}`);
  }
  await ctx.close();
}
await browser.close();
console.log(blocking ? `\n${blocking} critical/serious rule violation(s)` : "\nNo critical or serious accessibility problems");
process.exitCode = blocking ? 1 : 0;
