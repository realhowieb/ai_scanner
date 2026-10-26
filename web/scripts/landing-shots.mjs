// Landing-page captures (public/landing/*-current.jpg) of the signed-in app with fixed
// sample data: every /api/hsf call is answered here with synthetic values, so no live
// market data or real account appears. Runs against `npm run dev` (plain-http cookies)
// with a fake session cookie; the browser clock is pinned to 11:20 ET.
// Usage: BASE_URL=http://localhost:3100 CHROMIUM_PATH=... node scripts/landing-shots.mjs
// then convert OUT_DIR/*.png to JPEG in public/landing/.
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
const BASE = process.env.BASE_URL || "http://localhost:3200";
const ME = {"email": "sample@hsfinest.com", "name": "Sample", "plan": "premium", "plan_label": "Premium", "is_admin": false, "alert_limit": 25, "email_verified": true, "entitlements": {"can_scan_sp500": true, "can_scan_nasdaq": true, "can_premarket": true, "can_afterhours": true, "can_unusual_volume": true, "can_export_csv": true, "can_earnings": true, "can_ai_notes": true, "can_scan_history": true, "can_track_record": true, "can_early_breakout": true, "can_full_universe": true, "can_paper_trade": true, "can_email_alerts": true, "can_day_trader": true, "can_scan_alerts": true, "can_insider_alerts": true, "can_diagnostics": false, "can_admin_panel": false}, "pro_trial_until": null, "billing_attention": null};
const PREFS = {"scanner_views": [], "custom_views": [], "table": {"columns": ["setup", "spark", "last", "change", "gap", "rvol", "prob"], "compact": false}, "quiet_hours": {"enabled": false, "start": "22:00", "end": "07:00", "timezone": "America/New_York"}};
const DT_ROW = {"ticker": "AVBP", "open": 17.31, "last": 15.31, "change_dollar": -1.99, "chg_pct": -27.11, "gap_pct": -17.61, "vwap": 15.66, "vs_vwap_pct": -2.19, "rvol": 2.97, "volume": 412196, "adx": 32.3, "supertrend": 20.4, "supertrend_direction": "red", "ewo": -5.34, "previous_close": 21.01, "close_today": 15.31, "session_chg_pct": -27.13, "ext_chg_pct": null, "day_trade_score": 93.5, "dt_quality": "strong", "dt_direction": "bearish", "dt_reasons": ["Below VWAP (-2.23%)", "SuperTrend red", "EWO negative (-5.34)", "ADX 32 confirms trend", "RVOL 3.0x confirms participation", "Gap -17.6%"], "dt_conflicts": [], "quote_flags": [], "traded_today": null, "ema_cross": null, "trade_ts": "2026-10-26T19:59:58.084130313Z", "atr_pct": 12.97, "donchian_pos": 20, "donchian_breakout": null, "bb_pctb": 27, "bb_squeeze": false, "adx_period": 14, "adx_source": "alpaca_multi", "supertrend_period": 13, "supertrend_multiplier": 2, "supertrend_source": "alpaca_multi", "ewo_pct": -22.19, "ewo_fast": 5, "ewo_slow": 35, "ewo_source": "alpaca_multi"};
const FRESH = {"state": "fresh", "scan_state": "fresh", "market_data_state": "fresh", "last_successful_scan_at": "2026-10-26T19:39:19.731421+00:00", "scan_completed_at": "2026-10-26T19:39:13.349862+00:00", "market_data_at": "2026-10-26T19:35:56.557272+00:00", "expected_scan_at": "2026-10-26T19:35:00+00:00", "calendar_covered": true, "checked_at": "2026-10-26T22:00:46.008133+00:00", "timestamp_basis": "saved_scan", "market_data_note": null};
const OPP = {"ticker": "RRC", "score": 76, "score_version": "1.0", "score_components": {"signals_component": 24, "model_component": 38, "momentum_component": 14, "fading_penalty": 0}, "primary_setup": "Gapper", "n_signals": 2, "signals": ["gapper", "gainer"], "fading": false, "status": "STRONG", "breakout_score": 101.1001268743, "prob": 17.1, "prob_raw": 0.076370813, "chg_pct": 17.1187043148, "gap_pct": 11.3319421647, "last": 53.3300018311, "rvol": 10.1328399371, "prob_rank": 89, "previous_score": 76, "score_delta": 0, "movement_state": "UNCHANGED", "previous_status": "STRONG", "status_transition": null, "rank": 1, "previous_rank": 1};
const OUT = process.env.OUT_DIR || "landing-shots";
mkdirSync(OUT, { recursive: true });

const T = ["NVDA", "CRWD", "AMD", "APP", "MSFT", "AAPL", "SHOP", "PLTR"];
const SETUP = ["Breakout", "Breakout", "Gapper", "Breakout", "Golden cross", "Breakout", "Gainer", "Breakout"];
const SIG = { Breakout: ["breakout", "gainer"], Gapper: ["gapper", "gainer"], "Golden cross": ["golden_cross"], Gainer: ["gainer"] };
const now = new Date("2026-10-26T15:20:00Z"); // 11:20 ET, mid-session
const iso = (minAgo) => new Date(now.getTime() - minAgo * 60000).toISOString();
const fresh = { ...FRESH, last_successful_scan_at: iso(6), scan_completed_at: iso(6), market_data_at: iso(8), expected_scan_at: iso(10), checked_at: iso(0) };
const setups = T.map((t, i) => ({ ticker: t, score: 88 - 4 * i, primary_setup: SETUP[i], status: i < 4 ? "STRONG" : "DEVELOPING", n_signals: SIG[SETUP[i]].length,
  last: +(188.4 - 12 * i).toFixed(2), chg_pct: +(3.8 - 0.4 * i).toFixed(2), gap_pct: SETUP[i] === "Gapper" ? 2.1 : null, rvol: +(2.6 - 0.2 * i).toFixed(1),
  prob: +(41 - 3 * i).toFixed(1), signals: SIG[SETUP[i]], fading: false, breakout_score: +(72 - 4 * i).toFixed(1), prob_rank: 95 - 5 * i }));
const spark = (i, n = 21) => Array.from({ length: n }, (_, k) => +(100 + k * (1.2 - i * 0.08) + Math.sin(k * 1.3 + i) * 1.6).toFixed(2));

const me = ME;
const DT_T = ["NVDA", "AMD", "TSLA", "PLTR", "SMCI", "COIN", "META", "SNOW", "RIVN", "SOFI"];
const dtRow = DT_ROW;
const dtRows = DT_T.map((t, i) => {
  const bull = i % 4 !== 3; const chg = bull ? 4.6 - 0.35 * i : -(3.2 - 0.2 * i); const last = +(140 - 9 * i).toFixed(2);
  return { ...dtRow, ticker: t, last, open: +(last / (1 + chg / 200)).toFixed(2), chg_pct: +chg.toFixed(2), session_chg_pct: +chg.toFixed(2), change_dollar: +(last * chg / 100).toFixed(2),
    gap_pct: +(chg / 2).toFixed(2), vwap: +(last * (bull ? 0.99 : 1.01)).toFixed(2), vs_vwap_pct: bull ? 1.0 : -1.0, rvol: +(3.1 - 0.2 * i).toFixed(1), volume: 4200000 - 300000 * i,
    adx: 34 - i, supertrend_direction: bull ? "green" : "red", ewo: bull ? 2.4 : -2.1, ewo_pct: bull ? 8.1 : -7.2, previous_close: +(last / (1 + chg / 100)).toFixed(2), close_today: last,
    day_trade_score: +(91 - 4.5 * i).toFixed(1), dt_quality: i < 5 ? "strong" : "developing", dt_direction: bull ? "bullish" : "bearish",
    dt_reasons: bull ? ["Above VWAP (+1.0%)", "SuperTrend green", "EWO positive", `ADX ${34 - i} confirms trend`, `RVOL ${(3.1 - 0.2 * i).toFixed(1)}x confirms participation`] : ["Below VWAP (-1.0%)", "SuperTrend red", "EWO negative", `ADX ${34 - i} confirms trend`],
    dt_conflicts: [], quote_flags: [], trade_ts: iso(1), atr_pct: 4.2, donchian_pos: bull ? 88 : 14, bb_pctb: bull ? 84 : 18 };
});
const dt = { source: "movers", missing: 0, state: "regular", symbols: DT_T, as_of: iso(1), rows: dtRows,
  summary: { strong: 5, developing: 5, flagged: 0, missing: 0, best_long: "NVDA", best_short: "PLTR" } };
const brief = { available: true, has_previous_snapshot: true, freshness: fresh, snapshot_time: iso(6), phase: "regular",
  market: [{ label: "S&P 500 (SPY)", last: 612.4, chg_pct: 0.64 }, { label: "NASDAQ 100 (QQQ)", last: 538.9, chg_pct: 1.02 }],
  breadth: { advancers: 64, decliners: 36 },
  sectors: [["Tech", 1.48], ["Comm Svcs", 0.92], ["Cons Disc", 0.71], ["Industrials", 0.44], ["Financials", 0.21], ["Materials", 0.05], ["Health Care", -0.18], ["Energy", -0.36], ["Utilities", -0.62]].map(([sector, chg_pct]) => ({ sector, chg_pct })),
  opportunities: setups.slice(0, 6).map((s, i) => ({ ...OPP, ...s, previous_score: s.score - [2, 0, 3, -1, 0, 1][i], score_delta: [2, 0, 3, -1, 0, 1][i],
    movement_state: ["RISING", "UNCHANGED", "RISING", "FALLING", "UNCHANGED", "RISING"][i], previous_status: s.status, status_transition: null, rank: i + 1, previous_rank: [2, 1, 5, 3, 5, 7][i] })),
  gappers: [["AMD", 164.4, 3.0, 2.1], ["SHOP", 116.4, 1.4, 1.2], ["PLTR", 104.4, 1.0, 0.9]].map(([ticker, last, chg_pct, gap_pct]) => ({ ticker, last, chg_pct, gap_pct, earnings_days: null })),
  gainers: setups.slice(0, 5).map((s) => ({ ticker: s.ticker, chg_pct: s.chg_pct })),
  losers: [["INTC", -2.4], ["NKE", -1.9], ["BA", -1.6], ["PFE", -1.2], ["DIS", -0.9]].map(([ticker, chg_pct]) => ({ ticker, chg_pct })),
  golden_crosses: ["MSFT", "COST"], top_breakout_scores: setups.slice(0, 5).map((s) => ({ ticker: s.ticker, score: s.breakout_score })),
  prebreakout_picks: [["CRWD", 41.2], ["APP", 38.5], ["NOW", 35.9]].map(([ticker, prob]) => ({ ticker, prob, earnings_days: null })), prebreakout_locked: false,
  earnings_today: ["ORCL", "ADBE", "LULU"] };
const day = (d) => new Date(now.getTime() + d * 864e5).toISOString().slice(0, 10);
const earnings = [["ORCL", 0, "amc"], ["ADBE", 0, "amc"], ["LULU", 0, "amc"], ["COST", 1, "amc"], ["AVGO", 2, "amc"], ["FDX", 3, "bmo"]].map(([ticker, d, time]) => ({ ticker, earnings_date: day(d), days_until: d, time }));
const tape = { quotes: [["SPY", 612.4, 0.64], ["QQQ", 538.9, 1.02], ["DIA", 441.2, 0.31], ["IWM", 228.7, 0.88], ["NVDA", 188.4, 3.8], ["AAPL", 128.4, 1.8], ["MSFT", 140.4, 2.2]].map(([symbol, last, chg_pct]) => ({ symbol, last, chg_pct })), phase: "open" };

function answer(path, q) {
  if (path === "/v1/me") return me;
  if (path === "/v1/me/alerts/unread") return { count: 0, seen_at: iso(60) };
  if (path === "/v1/me/research-preferences") return PREFS;
  if (path === "/v1/market/tape") return tape;
  if (path === "/v1/scans/latest") return { freshness: fresh, scan_at: iso(6), total: setups.length, scan_total: setups.length, max_results: 25, limited: false, stale: false, setups };
  if (path === "/v1/sparklines") { const syms = (q.get("symbols") || "").split(","); return { last_bar: Object.fromEntries(syms.map((s) => [s, { close: setups.find((x) => x.ticker === s)?.last ?? 100, session: day(0) }])), series: Object.fromEntries(syms.map((s, i) => [s, spark(i)])) }; }
  if (path === "/v1/news/flags") return { flags: {}, hours: 24 };
  if (path === "/v1/day-trader") return dt;
  if (path === "/v1/day-trader/sparklines") { const syms = (q.get("symbols") || "").split(","); return { checked: syms, series: Object.fromEntries(syms.map((s, i) => [s, spark(i % 6, 60)])) }; }
  if (path === "/v1/brief") return brief;
  if (path === "/v1/earnings") return earnings;
  return null;
}

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
const misses = new Set();
for (const [suffix, viewport, scale] of [["", { width: 1425, height: 1000 }, 1], ["-mobile", { width: 390, height: 844 }, 2]]) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: scale });
  await ctx.addCookies([{ name: "hsf_rt", value: "sample", url: BASE }]);
  await ctx.route("**/api/hsf/**", async (route) => {
    const u = new URL(route.request().url());
    const path = u.pathname.replace(/^\/api\/hsf/, "");
    const body = answer(path, u.searchParams);
    if (body == null) { misses.add(path); return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Not found" }) }); }
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await ctx.route("**/ingest/**", (r) => r.fulfill({ status: 204, body: "" }));
  await ctx.clock.setFixedTime(now);
  const page = await ctx.newPage();
  for (const [name, path] of [["scanner", "/scanner"], ["day", "/day-trader"], ["brief", "/brief"]]) {
    await page.goto(BASE + path);
    await page.waitForLoadState("networkidle").catch(() => {});
    await page.waitForTimeout(2000);
    await page.addStyleTag({ content: "nextjs-portal{display:none!important} .tape-track{animation:none!important}" });
    if (!suffix) {
      const h = await page.evaluate(() => document.documentElement.scrollHeight);
      await page.setViewportSize({ width: viewport.width, height: Math.min(h, 2400) });
      await page.waitForTimeout(800);
    }
    await page.screenshot({ path: `${OUT}/${name}${suffix}-current.png` });
    if (!suffix) await page.setViewportSize(viewport);
    console.log("saved", name + suffix);
  }
  await ctx.close();
}
await browser.close();
console.log("unanswered:", [...misses].join(", ") || "none");
