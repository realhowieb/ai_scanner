// The daily workflow, end to end in a real browser through the BFF:
// sign in → find a setup → inspect the stock → save it to a watchlist → create an alert
// → sign out and back in ("return later") → check isolation → clean up.
//
// Everything it creates is named zz-e2e-<run id>; the alert uses a price threshold that
// cannot fire (999,999). It deletes only what it created. Credentials come from the
// environment and are never printed:
//   BASE_URL=http://localhost:3000 HSF_TEST_EMAIL=… HSF_TEST_PASSWORD_FILE=… \
//   [HSF_TEST_EMAIL_2=… (a second account, for isolation)] [OUT_DIR=screenshots] node scripts/journey.mjs
import { existsSync, mkdirSync, readFileSync } from "node:fs";
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
// Stock page buttons: "Watch ZS" / "Watch ZS elsewhere" and "Alert near $231.50" (#79 and later).
const SAVE_BTN = /^(Watch \S+( elsewhere)?|Save to watchlist)$/;
const ALERT_BTN = /^(Alert near .*|Set price alert)$/;
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
    let where = "";
    try { where = ` (at ${new URL(page.url()).pathname})`; } catch { /* page closed */ }
    check(name, false, String(e.message || e).split("\n")[0].slice(0, 200) + where);
  }
};

// CHROMIUM_PATH, else the cloud container's browser, else Playwright's own download
// (`npx playwright-core install chromium`, as the Live acceptance workflow does).
const LOCAL_CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const executablePath = process.env.CHROMIUM_PATH || (existsSync(LOCAL_CHROMIUM) ? LOCAL_CHROMIUM : undefined);
const browser = await chromium.launch({ executablePath });
const problems = [];

async function signIn(ctx, who, pw) {
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" && !/status of 4\d\d/.test(m.text())) problems.push(`console: ${m.text()}`); });
  await page.goto(`${BASE}/today`);
  await page.waitForURL(/\/login/);
  await page.getByLabel("Email").fill(who);
  await page.getByLabel("Password", { exact: true }).fill(pw);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(/\/today/, { timeout: 90_000 });
  return page;
}

// Deletes zz-e2e-* watchlists and the 999,999 price alerts that an earlier run failed to
// clean up (a leftover duplicate alert would make "create an alert" fail every later run).
// Goes through the BFF with the signed-in page's session; it never fails the run.
async function sweepLeftovers(p, label) {
  const removed = await p.evaluate(async (threshold) => {
    const get = async (path) => (await fetch(`/api/hsf${path}`)).json();
    const del = (path) => fetch(`/api/hsf${path}`, { method: "DELETE" });
    let n = 0;
    const alerts = (await get("/v1/alerts")).alerts || [];
    for (const a of alerts) if (a.type === "price" && Number(a.threshold) >= threshold) { await del(`/v1/alerts/${a.id}`); n++; }
    const lists = await get("/v1/watchlists");
    for (const w of Array.isArray(lists) ? lists : []) if (String(w.name || "").startsWith("zz-e2e-")) { await del(`/v1/watchlists/${w.id}`); n++; }
    return n;
  }, Number(THRESHOLD)).catch((e) => `error: ${String(e.message || e).slice(0, 120)}`);
  console.log(`[INFO] ${label}: removed ${removed} leftover test records`);
}
// The Alerts page opens on Activity; saved alerts live under the Rules tab.
const rulesTab = (p) => p.getByRole("tab", { name: "Rules" }).click();
const shot = (page, name) => page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });

const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
let page = await signIn(ctx, email, password);
check("sign in through the BFF", true);
await sweepLeftovers(page, "before the run");
let ticker = null;
let watchlistUrl = null;

await step("Top setups lists stocks only, ETFs on their own line", async () => {
  await page.goto(`${BASE}/today`);
  const card = page.getByRole("region", { name: "Top setups" });
  await card.getByRole("heading", { name: "Top setups" }).waitFor();
  const line = card.getByTestId("etf-line");
  if (!(await line.count())) {
    check("Top setups lists stocks only, ETFs on their own line", true, "no ETFs ranked in this scan");
    return;
  }
  const href = (l) => decodeURIComponent((l ?? "").split("/stocks/")[1] ?? "");
  const etfs = (await line.locator('a[href^="/stocks/"]').evaluateAll((as) => as.map((a) => a.getAttribute("href")))).map(href);
  const board = (await card.locator('table a[href^="/stocks/"]').evaluateAll((as) => as.map((a) => a.getAttribute("href")))).map(href);
  const overlap = etfs.filter((t) => board.includes(t));
  const all = await line.getByRole("link", { name: "All ETFs on the Scanner" }).getAttribute("href");
  check("Top setups lists stocks only, ETFs on their own line", overlap.length === 0 && all === "/scanner?sector=ETF",
        `ETFs ${etfs.join(",")}${overlap.length ? `; also on the board: ${overlap.join(",")}` : ""}`);
});

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
  // The ticker from the link itself: the cell also carries a logo letter and the company name.
  ticker = decodeURIComponent(((await first.getAttribute("href")) ?? "").split("/stocks/")[1] ?? "").trim();
  await page.getByRole("button", { name: `Save ${ticker} to a watchlist` }).first().click();
  const dlg = page.getByRole("dialog", { name: `Save ${ticker} to a watchlist` });
  await dlg.getByLabel("New watchlist").fill(RUN);
  await dlg.getByRole("button", { name: "Create and save" }).click();
  await dlg.getByText(`Saved ${ticker} to ${RUN}.`).waitFor();
  await shot(page, "journey-1-scanner-save");
  await dlg.getByRole("button", { name: "Done" }).click();
  check("find a setup and save it to a new watchlist from the Scanner", true, ticker);
});

await step("Scanner quick view opens a setup without leaving the page", async () => {
  await page.getByRole("button", { name: `Quick view ${ticker}` }).first().click();
  const panel = page.getByRole("dialog", { name: ticker });
  await panel.getByRole("link", { name: "Open full page" }).waitFor();
  await page.keyboard.press("Escape");
  await panel.waitFor({ state: "hidden" });
  check("Scanner quick view opens a setup without leaving the page", /\/scanner/.test(page.url()), ticker);
});

await step("Scanner shows the market condition (Trending up / Choppy / Risk-off)", async () => {
  // GET /v1/market/condition; the badge is left out only when SPY prices can't be loaded.
  const badge = page.getByRole("note", { name: /^Market condition: (Trending up|Choppy|Risk-off)$/ });
  try {
    await badge.waitFor({ timeout: 8000 });
  } catch {
    check("Scanner shows the market condition (Trending up / Choppy / Risk-off)", false, "no badge (SPY closes unavailable?)");
    return;
  }
  check("Scanner shows the market condition (Trending up / Choppy / Risk-off)", true, await badge.getAttribute("aria-label"));
});

await step("Scanner shows when HSF picked up each setup", async () => {
  // "Picked up" column: the first scan of the stock's current run, with the price then (pickup in /v1/scans/latest).
  await page.getByRole("columnheader", { name: /Picked up/ }).first().waitFor();
  const when = ((await page.locator("table .picked-cell .mono").first().textContent()) ?? "").trim();
  check("Scanner shows when HSF picked up each setup", / ET$/.test(when), when);
});

await step("Scanner shows relative strength vs SPY and filters to stocks beating SPY", async () => {
  // "vs SPY" column: 20-session return minus SPY's (rs_20d), with an RS high badge; "—" on rows
  // from scans saved before relative strength existed. The vs SPY filter sends beating_spy.
  await page.getByRole("columnheader", { name: /vs SPY/ }).first().waitFor();
  const first = ((await page.locator("table .vs-spy .mono").first().textContent({ timeout: 5_000 }).catch(() => "—")) ?? "").trim();
  await page.getByRole("combobox", { name: "vs SPY" }).selectOption("beat");
  await page.waitForURL(/[?&]spy=beat/);
  await page.locator("table .tk, .empty").first().waitFor();
  const shown = await page.locator("table .vs-spy .mono").allTextContents();
  const allBeat = shown.every((t) => t.trim().startsWith("+"));
  await page.getByRole("combobox", { name: "vs SPY" }).selectOption("");
  await page.waitForURL((u) => !u.search.includes("spy=beat"));
  await page.locator("table .tk").first().waitFor();
  check("Scanner shows relative strength vs SPY and filters to stocks beating SPY", allBeat, `first row ${first}; ${shown.length} rows beating SPY`);
});

await step("Scanner phone cards keep ticker, badges and actions inside the card; Compare toggles", async () => {
  // Same session at iPhone width: the table turns into cards (#ticker row, badge row, fact grid).
  await page.setViewportSize({ width: 390, height: 844 });
  const card = page.locator(".scan-card").first();
  await card.waitFor();
  const over = await page.$$eval(".scan-card", (cs) => cs.filter((c) => {
    const r = c.getBoundingClientRect();
    return [...c.querySelectorAll("*")].some((e) => { const q = e.getBoundingClientRect(); return q.width > 0 && q.right > r.right + 1; });
  }).length);
  const chip = card.getByRole("button", { name: /^Compare / });
  await chip.click();
  const pressed = await chip.getAttribute("aria-pressed");
  await chip.click();
  await shot(page, "journey-1b-scanner-phone");
  await page.setViewportSize({ width: 1440, height: 1000 });
  check("Scanner phone cards keep ticker, badges and actions inside the card; Compare toggles", over === 0 && pressed === "true", `overflowing cards ${over}, compare ${pressed}`);
});

await step("Day Trader shows each name's move vs SPY today", async () => {
  // vs SPY = Chg % minus SPY's in the same snapshot (vs_spy_pct). A source with no rows shows no table.
  await page.goto(`${BASE}/day-trader`);
  await page.locator(".dt-table, .dt-cards, .empty").first().waitFor({ timeout: 60_000 });
  const table = await page.locator(".dt-table").count();
  const header = table ? await page.getByRole("columnheader", { name: /vs SPY/ }).count() : 0;
  await shot(page, "journey-1c-day-trader-vs-spy");
  await page.goto(`${BASE}/scanner`);
  await page.locator("table .tk").first().waitFor();
  check("Day Trader shows each name's move vs SPY today", !table || header > 0, table ? `vs SPY header ${header}` : "no rows right now");
});

await step("Stock Intelligence shows the watchlist; saving again reports already present", async () => {
  await page.locator(`table a.tk[href="/stocks/${encodeURIComponent(ticker)}"]`).first().click();
  await page.waitForURL(new RegExp(`/stocks/${ticker}`));
  await page.getByRole("heading", { name: "HSF Score", exact: true }).waitFor();
  await page.getByRole("link", { name: RUN }).waitFor();
  await page.getByRole("button", { name: SAVE_BTN }).click();
  const dlg = page.getByRole("dialog");
  await dlg.getByRole("button", { name: new RegExp(RUN) }).waitFor();
  const label = await dlg.getByRole("button", { name: new RegExp(RUN) }).textContent();
  await dlg.getByRole("button", { name: "Done" }).click();
  check("Stock Intelligence shows the watchlist; saving again reports already present", /Already in it/.test(label), label.replace(/\s+/g, " "));
});

await step("create a non-firing price alert from Stock Intelligence", async () => {
  await page.getByRole("button", { name: ALERT_BTN }).click();
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
  await page.getByRole("button", { name: ALERT_BTN }).click();
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
  await page.getByRole("button", { name: "Add tickers" }).click(); // the add form opens from a button once the list has tickers
  await page.getByLabel("Add tickers").fill(`${ticker} NOT_A_TICKER!!`);
  await page.getByRole("button", { name: "Add", exact: true }).click();
  const status = await page.getByRole("status").filter({ hasText: /Already in the list|Not valid/ }).first().textContent();
  await shot(page, "journey-4-watchlist");
  check("watchlist: ticker listed, note saved, invalid and duplicate tickers reported",
    status.includes(`Already in the list: ${ticker}`) && status.includes("Not valid"), status.slice(0, 160));
});

await step("watchlist summary chips filter the list", async () => {
  const chip = page.getByRole("button", { name: /^1 with notes/ });
  await chip.click();
  const note = await page.locator(".wl-filter-note").textContent();
  const pressed = await chip.getAttribute("aria-pressed");
  await chip.click(); // clicking again shows every ticker
  await page.locator(".wl-filter-note").waitFor({ state: "detached" });
  check("watchlist summary chips filter the list", pressed === "true" && /Showing 1 of 1: with notes/.test(note), note.replace(/\s+/g, " "));
});

await step("watchlist table sorts by a column, selects rows and switches tabs", async () => {
  await page.getByRole("button", { name: "Sort by Ticker" }).click();
  const sorted = await page.getByRole("columnheader", { name: /Ticker/ }).getAttribute("aria-sort");
  await page.getByRole("checkbox", { name: `Select ${ticker}` }).check();
  const bar = await page.getByRole("region", { name: "Selected tickers" }).textContent();
  await page.getByRole("checkbox", { name: `Select ${ticker}` }).uncheck();
  await page.getByRole("tab", { name: "Performance" }).click();
  await page.getByRole("heading", { name: "How your stocks did" }).waitFor();
  await page.getByRole("tab", { name: "Tickers" }).click();
  check("watchlist table sorts by a column, selects rows and switches tabs", sorted === "ascending" && /1 selected/.test(bar), `sort ${sorted}; ${bar.replace(/\s+/g, " ").slice(0, 60)}`);
});

await step("Scanner Watchlist tab shows the watchlist's stocks", async () => {
  const id = new URL(watchlistUrl).searchParams.get("id");
  await page.goto(`${BASE}/scanner?signal=watchlist&wl=${id}`);
  await page.getByRole("button", { name: "Watchlist", exact: true, pressed: true }).waitFor();
  const summary = await page.getByTestId("watchlist-scan-summary").textContent();
  await page.locator(`table a.tk[href="/stocks/${encodeURIComponent(ticker)}"]`).first().waitFor();
  await shot(page, "journey-4b-scanner-watchlist");
  check("Scanner Watchlist tab shows the watchlist's stocks", summary.includes(RUN), summary.replace(/\s+/g, " ").slice(0, 160));
});

await step("alerts page: capacity, turn off", async () => {
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: /^Alerts(, \d+ new)?$/ }).click();
  await page.waitForURL(/\/alerts/);
  await rulesTab(page);
  await page.getByText(`${ticker} price rises above $${SHOWN}`).waitFor();
  const capacity = await page.getByLabel("Alert summary").textContent();
  await page.getByRole("button", { name: `Turn off: ${ticker} price rises above $${SHOWN}` }).click();
  await page.getByRole("button", { name: `Turn on: ${ticker} price rises above $${SHOWN}` }).waitFor();
  await shot(page, "journey-5-alerts");
  check("alerts page: capacity, turn off", true, capacity);
});

await step("alerts summary chip opens Rules filtered to paused; Activity table or empty state shows", async () => {
  await page.getByLabel("Alert summary").getByRole("button", { name: /^(\d+|—) paused$/ }).click();
  const pausedOn = await page.getByRole("radio", { name: /^Paused/ }).isChecked();
  await page.getByRole("region", { name: "Your alert rules" }).getByText(`${ticker} price rises above $${SHOWN}`).waitFor();
  await page.getByRole("tab", { name: "Activity" }).click();
  await page.getByRole("region", { name: "Fired alerts" }).or(page.getByText("No recent triggers.")).or(page.getByRole("status").filter({ hasText: /^No alerts/ })).first().waitFor();
  await shot(page, "journey-5b-alerts-activity");
  check("alerts summary chip opens Rules filtered to paused; Activity table or empty state shows", pausedOn);
});

await step("return later: sign out, sign in, everything is still there", async () => {
  await page.getByRole("button", { name: /Account/ }).click();
  await page.getByRole("menuitem", { name: "Sign out", exact: true }).click();
  await page.waitForURL(/\/signed-out/);
  await page.getByRole("heading", { name: "You're signed out" }).waitFor();
  await page.goto(`${BASE}/alerts`);
  await page.waitForURL(/\/login/);
  await page.close();
  page = await signIn(ctx, email, password);
  await page.goto(watchlistUrl);
  await page.getByText("e2e note: watch the breakout level").waitFor();
  await page.goto(`${BASE}/alerts`);
  await rulesTab(page);
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
  await rulesTab(mp);
  await mp.getByText(`${ticker} price rises above $${SHOWN}`).waitFor();
  await shot(mp, "journey-m-alerts");
  await mp.goto(`${BASE}/stocks/${ticker}`);
  await mp.getByRole("button", { name: SAVE_BTN }).click();
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
    await rulesTab(op);
    await op.getByRole("heading", { name: "Your alerts" }).waitFor();
    const sawAlert = await op.getByText(`$${SHOWN}`).count();
    await other.close();
    check("another account can't see or open these", sawList === 0 && sawAlert === 0, `list seen ${sawList}, alert seen ${sawAlert}`);
  });
}

// Clean up from a fresh sign-in, so a step that failed half way (signed out, closed page)
// doesn't also leave this run's records behind.
await step("public track record and market explainer load signed out", async () => {
  const anon = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const ap = await anon.newPage();
  await ap.goto(`${BASE}/results`);
  await ap.getByRole("heading", { level: 1, name: "Every top pick, and how it did" }).waitFor();
  const verdict = ap.getByRole("heading", { name: /^(No proven edge yet|Ahead of the S&P 500 so far|Behind the S&P 500 so far)$/ });
  const ok = await verdict.waitFor({ timeout: 15000 }).then(() => true, () => false);
  const verdictText = ok ? await verdict.textContent() : "no verdict (API answered nothing?)";
  const rows = ok ? await ap.locator("#picks-h ~ .table-wrap tbody tr").count() : 0;
  const overflow = await ap.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  await shot(ap, "journey-results-phone");
  await ap.goto(`${BASE}/how-hsf-works#method-market`);
  const market = await ap.locator("#method-market").getByRole("heading", { name: "Market conditions" }).count();
  const speed = await ap.locator("#method-freshness").count();
  await anon.close();
  check("public track record and market explainer load signed out", ok && rows > 0 && !overflow && market === 1 && speed === 1,
    `${verdictText}; ${rows} picks listed${overflow ? "; page scrolls sideways on a phone" : ""}`);
});

await step("sign-up form is short and keeps a plan picked on Pricing", async () => {
  const anon = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const ap = await anon.newPage();
  await ap.goto(`${BASE}/signup?plan=pro&interval=month`);
  await ap.getByText(/You picked Pro/).waitFor();
  const fields = { email: await ap.getByLabel("Email").count(), password: await ap.getByLabel("Password", { exact: true }).count(),
    name: await ap.getByLabel("Display name").count(), repeat: await ap.getByLabel("Repeat password").count() };
  const intent = await ap.evaluate(() => localStorage.getItem("hsf-plan-intent") || "");
  await anon.close();
  check("sign-up form is short and keeps a plan picked on Pricing",
    fields.email === 1 && fields.password === 1 && fields.name === 0 && fields.repeat === 0 && intent.includes('"pro"'),
    `fields ${JSON.stringify(fields)}`);
});

const cleanCtx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
await step("clean up: delete the alert and the watchlist (confirmed)", async () => {
  page = await signIn(cleanCtx, email, password);
  if (!ticker || !watchlistUrl) throw new Error("nothing to clean up: an earlier step failed before creating it");
  await page.goto(`${BASE}/alerts`);
  await rulesTab(page);
  await page.getByRole("button", { name: `Delete: ${ticker} price rises above $${SHOWN}` }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete alert" }).click();
  await page.getByRole("button", { name: `Delete: ${ticker} price rises above $${SHOWN}` }).waitFor({ state: "detached" });
  await page.goto(watchlistUrl);
  await page.getByRole("heading", { name: new RegExp(RUN) }).waitFor();
  await page.getByText("Manage list").click(); // Delete sits under the Manage list disclosure
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete watchlist" }).click();
  await page.getByText(RUN).first().waitFor({ state: "detached", timeout: 10_000 }).catch(() => {});
  const left = await page.getByText(RUN).count();
  check("clean up: delete the alert and the watchlist (confirmed)", left === 0, `${left} references left`);
});

if (!page.isClosed()) await sweepLeftovers(page, "after the run");
await browser.close();
const failed = results.filter((r) => !r.ok).length;
if (problems.length) console.log("BROWSER PROBLEMS:\n" + problems.join("\n"));
console.log(`\n${results.length - failed} passed, ${failed} failed (run ${RUN})`);
process.exitCode = failed || problems.length ? 1 : 0;
