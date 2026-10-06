// Screenshots of the main screens at desktop and phone sizes, against a running
// Web v2 server (BASE_URL) signed in with a test account. Credentials come from the
// environment (HSF_TEST_EMAIL, HSF_TEST_PASSWORD_FILE) and are never printed.
// Usage: BASE_URL=http://localhost:3100 HSF_TEST_EMAIL=... HSF_TEST_PASSWORD_FILE=... npm run screenshots
import { readFileSync, mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL || "http://localhost:3100";
const OUT = process.env.OUT_DIR || "screenshots";
const PREFIX = process.env.PREFIX || "";
const email = process.env.HSF_TEST_EMAIL;
const password = readFileSync(process.env.HSF_TEST_PASSWORD_FILE, "utf8").trim();
const PAGES = (process.env.PAGES || "today,scanner,stock,custom").split(",");
const STOCK = process.env.STOCK || "AAPL";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
const problems = [];
for (const [size, viewport] of [["desktop", { width: 1440, height: 1000 }], ["mobile", { width: 390, height: 844 }]]) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: size === "mobile" ? 2 : 1 });
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") problems.push(`${size} console: ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`${size} pageerror: ${e.message}`));
  await page.goto(`${BASE}/today`);
  await page.waitForURL(/\/login/);
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(/\/today/);
  const cookies = await ctx.cookies();
  if (cookies.some((c) => c.name.includes("hsf_") && !c.httpOnly)) problems.push("a session cookie is readable by scripts");
  const readable = await page.evaluate(() => document.cookie + JSON.stringify(localStorage) + JSON.stringify(sessionStorage));
  if (/eyJ|refresh/i.test(readable)) problems.push("a token is visible to page scripts");

  const shots = {
    today: async () => { await page.goto(`${BASE}/today`); await page.getByRole("heading", { name: "Top setups" }).waitFor(); },
    scanner: async () => { await page.goto(`${BASE}/scanner`); await page.locator(".tk:visible").first().waitFor(); },
    stock: async () => { await page.goto(`${BASE}/stocks/${STOCK}`); await page.getByRole("heading", { name: "HSF Score" }).waitFor(); },
    custom: async () => {
      await page.goto(`${BASE}/scanner/custom`);
      const start = page.getByRole("button", { name: "Start scan" });
      await start.waitFor();
      await start.click();
      await page.getByText(/Results ·/).waitFor({ timeout: 180_000 });
    },
  };
  for (const name of PAGES) {
    await shots[name]();
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${OUT}/${PREFIX}${name}-${size}.png`, fullPage: true });
    console.log(`saved ${PREFIX}${name}-${size}.png`);
  }
  await ctx.close();
}
await browser.close();
if (problems.length) {
  console.log("PROBLEMS:\n" + problems.join("\n"));
  process.exitCode = 1;
}
