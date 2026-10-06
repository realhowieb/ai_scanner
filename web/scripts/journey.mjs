// The daily workflow, end to end in a real browser through the BFF:
// sign in → find a setup → inspect the stock → save it to a watchlist → create an alert
// → sign out and back in ("return later") → check isolation → clean up.
//
// Everything it creates is named zz-e2e-<run id>; the alert uses a price threshold that
// cannot fire (999,999). It deletes only what it created. Credentials come from the
// environment and are never printed:
//   BASE_URL=http://localhost:3000 HSF_TEST_EMAIL=… HSF_TEST_PASSWORD_FILE=… \
//   [HSF_TEST_EMAIL_2=… (a second account, for isolation)] [OUT_DIR=screenshots] node scripts/journey.mjs
import { mkdirSync, readFileSync } from "node:fs";
import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL || "http://localhost:3100";
const OUT = process.env.OUT_DIR || "screenshots";
const email = process.env.HSF_TEST_EMAIL;
const email2 = process.env.HSF_TEST_EMAIL_2;
const password = readFileSync(process.env.HSF_TEST_PASSWORD_FILE, "utf8").trim();
const password2 = process.env.HSF_TEST_PASSWORD_FILE_2 ? readFileSync(process.env.HSF_TEST_PASSWORD_FILE_2, "utf8").trim() : password;
const RUN = `zz-e2e-${Date.now().toString(36)}`;
const THRESHOLD = "999999";
const SHOWN = "999,999"; // as the app formats it
mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, evidence = "") => {
  results.push({ name, ok });
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${evidence ? ` — ${evidence}` : ""}`);
};
const step = async (name, fn) => {
  try {
    await fn();
  } catch (e) {
    check(name, false, String(e.message || e).split("\n")[0].slice(0, 200));
  }
};

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
const problems = [];

async function signIn(ctx, who, pw) {
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" && !/status of 4\d\d/.test(m.text())) problems.push(`console: ${m.text()}`); });
  await page.goto(`${BASE}/today`);
  await page.waitForURL(/\/login/);
  await page.getByLabel("Email").fill(who);
  await page.getByLabel("Password").fill(pw);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(/\/today/, { timeout: 90_000 });
  return page;
}
const shot = (page, name) => page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });

const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
let page = await signIn(ctx, email, password);
check("sign in through the BFF", true);
let ticker = null;
let watchlistUrl = null;

await step("session survives reload and navigation", async () => {
  await page.reload();
  await page.getByRole("heading", { name: "Top setups" }).waitFor();
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Scanner", exact: true }).click();
  await page.waitForURL(/\/scanner$/);
  check("session survives reload and navigation", true);
});

await step("find a setup and save it to a new watchlist from the Scanner", async () => {
  const first = page.locator("table .tk").first();
  await first.waitFor();
  ticker = (await first.textContent()).trim();
  await page.getByRole("button", { name: `Save ${ticker} to a watchlist` }).first().click();
  const dlg = page.getByRole("dialog", { name: `Save ${ticker} to a watchlist` });
  await dlg.getByLabel("New watchlist").fill(RUN);
  await dlg.getByRole("button", { name: "Create and save" }).click();
  await dlg.getByText(`Saved ${ticker} to ${RUN}.`).waitFor();
  await shot(page, "journey-1-scanner-save");
  await dlg.getByRole("button", { name: "Done" }).click();
  check("find a setup and save it to a new watchlist from the Scanner", true, ticker);
});

await step("Stock Intelligence shows the watchlist; saving again reports already present", async () => {
  await page.locator("table").getByRole("link", { name: ticker, exact: true }).first().click();
  await page.waitForURL(new RegExp(`/stocks/${ticker}`));
  await page.getByRole("heading", { name: "HSF Score" }).waitFor();
  await page.getByRole("link", { name: RUN }).waitFor();
  await page.getByRole("button", { name: "Save to watchlist" }).click();
  const dlg = page.getByRole("dialog");
  await dlg.getByRole("button", { name: new RegExp(RUN) }).waitFor();
  const label = await dlg.getByRole("button", { name: new RegExp(RUN) }).textContent();
  await dlg.getByRole("button", { name: "Done" }).click();
  check("Stock Intelligence shows the watchlist; saving again reports already present", /Already in it/.test(label), label.replace(/\s+/g, " "));
});

await step("create a non-firing price alert from Stock Intelligence", async () => {
  await page.getByRole("button", { name: "Set price alert" }).click();
  const dlg = page.getByRole("dialog", { name: `Price alert for ${ticker}` });
  await dlg.getByLabel("Price ($)").fill(THRESHOLD);
  await shot(page, "journey-2-price-alert");
  await dlg.getByRole("button", { name: "Create alert" }).click();
  await dlg.getByText(/Created:/).waitFor();
  await dlg.getByRole("button", { name: "Done" }).click();
  await page.getByText(`${ticker} price rises above $${SHOWN}`).waitFor();
  await shot(page, "journey-3-stock");
  check("create a non-firing price alert from Stock Intelligence", true);
});

await step("duplicate alert is refused with the server's message", async () => {
  await page.getByRole("button", { name: "Set price alert" }).click();
  const dlg = page.getByRole("dialog");
  await dlg.getByLabel("Price ($)").fill(THRESHOLD);
  await dlg.getByRole("button", { name: "Create alert" }).click();
  const err = dlg.getByRole("alert").first();
  await err.waitFor();
  const text = await err.textContent();
  await dlg.getByRole("button", { name: "Close" }).click();
  check("duplicate alert is refused with the server's message", /already/i.test(text), text.slice(0, 120));
});

await step("watchlist: ticker listed, note saved, invalid and duplicate tickers reported", async () => {
  await page.getByRole("link", { name: RUN }).click();
  await page.waitForURL(/\/watchlists\?id=\d+/);
  watchlistUrl = page.url();
  await page.getByRole("heading", { name: new RegExp(RUN) }).waitFor();
  await page.getByRole("button", { name: `Add note for ${ticker}` }).click();
  await page.getByLabel(`Note for ${ticker}`).fill("e2e note: watch the breakout level");
  await page.getByRole("button", { name: "Save note" }).click();
  await page.getByText("e2e note: watch the breakout level").waitFor();
  await page.getByLabel("Add tickers").fill(`${ticker} NOT_A_TICKER!!`);
  await page.getByRole("button", { name: "Add", exact: true }).click();
  const status = await page.getByRole("status").filter({ hasText: /Already in the list|Not valid/ }).first().textContent();
  await shot(page, "journey-4-watchlist");
  check("watchlist: ticker listed, note saved, invalid and duplicate tickers reported",
    status.includes(`Already in the list: ${ticker}`) && status.includes("Not valid"), status.slice(0, 160));
});

await step("alerts page: capacity, turn off", async () => {
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Alerts", exact: true }).click();
  await page.waitForURL(/\/alerts/);
  await page.getByText(`${ticker} price rises above $${SHOWN}`).waitFor();
  const capacity = await page.getByText(/Using \d+ of \d+ on your/).textContent();
  await page.getByRole("button", { name: `Turn off: ${ticker} price rises above $${SHOWN}` }).click();
  await page.getByRole("button", { name: `Turn on: ${ticker} price rises above $${SHOWN}` }).waitFor();
  await shot(page, "journey-5-alerts");
  check("alerts page: capacity, turn off", true, capacity);
});

await step("return later: sign out, sign in, everything is still there", async () => {
  await page.getByRole("button", { name: /Account/ }).click();
  await page.getByRole("menuitem", { name: "Sign out" }).click();
  await page.waitForURL(/\/login/);
  await page.goto(`${BASE}/alerts`);
  await page.waitForURL(/\/login/);
  await page.close();
  page = await signIn(ctx, email, password);
  await page.goto(watchlistUrl);
  await page.getByText("e2e note: watch the breakout level").waitFor();
  await page.goto(`${BASE}/alerts`);
  await page.getByRole("button", { name: `Turn on: ${ticker} price rises above $${SHOWN}` }).waitFor();
  check("return later: sign out, sign in, everything is still there", true);
});

await step("phone layout: watchlist, alerts and the save dialog", async () => {
  const m = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
  const mp = await signIn(m, email, password);
  await mp.goto(watchlistUrl);
  await mp.getByText("e2e note: watch the breakout level").waitFor();
  await shot(mp, "journey-m-watchlist");
  await mp.goto(`${BASE}/alerts`);
  await mp.getByText(`${ticker} price rises above $${SHOWN}`).waitFor();
  await shot(mp, "journey-m-alerts");
  await mp.goto(`${BASE}/stocks/${ticker}`);
  await mp.getByRole("button", { name: "Save to watchlist" }).click();
  await mp.getByRole("dialog").getByRole("button", { name: new RegExp(RUN) }).waitFor();
  await mp.screenshot({ path: `${OUT}/journey-m-save-dialog.png` });
  const width = await mp.evaluate(() => document.documentElement.scrollWidth);
  await m.close();
  check("phone layout: watchlist, alerts and the save dialog", width <= 390, `page width ${width}px`);
});

if (email2) {
  await step("another account can't see or open these", async () => {
    const other = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const op = await signIn(other, email2, password2);
    await op.goto(watchlistUrl);
    await op.getByRole("heading", { name: "Watchlists" }).waitFor();
    await op.waitForTimeout(1500);
    const sawList = await op.getByText(RUN).count();
    await op.goto(`${BASE}/alerts`);
    await op.getByRole("heading", { name: "Your alerts" }).waitFor();
    const sawAlert = await op.getByText(`$${SHOWN}`).count();
    await other.close();
    check("another account can't see or open these", sawList === 0 && sawAlert === 0, `list seen ${sawList}, alert seen ${sawAlert}`);
  });
}

await step("clean up: delete the alert and the watchlist (confirmed)", async () => {
  await page.goto(`${BASE}/alerts`);
  await page.getByRole("button", { name: `Delete: ${ticker} price rises above $${SHOWN}` }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete alert" }).click();
  await page.getByRole("button", { name: `Delete: ${ticker} price rises above $${SHOWN}` }).waitFor({ state: "detached" });
  await page.goto(watchlistUrl);
  await page.getByRole("heading", { name: new RegExp(RUN) }).waitFor();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete watchlist" }).click();
  await page.getByText(RUN).first().waitFor({ state: "detached", timeout: 10_000 }).catch(() => {});
  const left = await page.getByText(RUN).count();
  check("clean up: delete the alert and the watchlist (confirmed)", left === 0, `${left} references left`);
});

await browser.close();
const failed = results.filter((r) => !r.ok).length;
if (problems.length) console.log("BROWSER PROBLEMS:\n" + problems.join("\n"));
console.log(`\n${results.length - failed} passed, ${failed} failed (run ${RUN})`);
process.exitCode = failed || problems.length ? 1 : 0;
