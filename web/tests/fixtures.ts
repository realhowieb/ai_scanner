// Test fixtures only. The running app always uses live API data.
import type { Schemas } from "@/api/client";

const FEATURES = ["can_scan_sp500", "can_scan_nasdaq", "can_premarket", "can_afterhours", "can_unusual_volume", "can_export_csv",
  "can_earnings", "can_ai_notes", "can_scan_history", "can_track_record", "can_early_breakout", "can_full_universe",
  "can_paper_trade", "can_email_alerts", "can_day_trader", "can_diagnostics", "can_admin_panel"];
const PRO = new Set(["can_scan_sp500", "can_scan_nasdaq", "can_premarket", "can_afterhours", "can_unusual_volume", "can_export_csv",
  "can_earnings", "can_scan_history", "can_track_record", "can_email_alerts", "can_day_trader"]);
const PREMIUM = new Set([...PRO, "can_ai_notes", "can_early_breakout", "can_full_universe", "can_paper_trade"]);

export function me(plan: "basic" | "pro" | "premium"): Schemas["Me"] {
  const on = plan === "basic" ? new Set(["can_scan_sp500"]) : plan === "pro" ? PRO : PREMIUM;
  return {
    email: `${plan}@example.invalid`, name: plan, plan, plan_label: { basic: "Free", pro: "Pro", premium: "Premium" }[plan],
    is_admin: false, alert_limit: 1, email_verified: true,
    entitlements: Object.fromEntries(FEATURES.map((f) => [f, on.has(f)])),
  };
}

export function setup(ticker: string, score: number, extra: Partial<Schemas["ScanSetup"]> = {}): Schemas["ScanSetup"] {
  return { ticker, score, primary_setup: "Breakout", status: "STRONG", n_signals: 2, last: 50, chg_pct: 2, gap_pct: null, rvol: 1.5,
    prob: null, signals: ["breakout"], fading: false, breakout_score: 80, ...extra };
}
