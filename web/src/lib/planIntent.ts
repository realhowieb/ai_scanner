// The paid plan a visitor picked on Pricing before they had an account. Kept through
// sign-up (and email confirmation, which may happen in another tab) so they land on that
// plan's checkout instead of hunting for the upgrade button. Browser storage only, for a
// week; nothing breaks without it.
import type { BillingInterval } from "@/components/BillingInterval";

export type PaidPlan = "pro" | "premium";
export type PlanIntent = { plan: PaidPlan; interval: BillingInterval; at: number };

const KEY = "hsf-plan-intent";
const KEEP_MS = 7 * 24 * 3600 * 1000;

export function parsePlan(raw: string | null | undefined): PaidPlan | null {
  return raw === "pro" || raw === "premium" ? raw : null;
}

export function parseInterval(raw: string | null | undefined): BillingInterval {
  return raw === "year" ? "year" : "month";
}

export function savePlanIntent(plan: PaidPlan, interval: BillingInterval, now = Date.now()): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ plan, interval, at: now }));
  } catch {
    /* private mode */
  }
}

export function readPlanIntent(now = Date.now()): PlanIntent | null {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "null") as Partial<PlanIntent> | null;
    const plan = parsePlan(raw?.plan);
    if (plan && typeof raw?.at === "number" && now - raw.at < KEEP_MS) return { plan, interval: parseInterval(raw.interval), at: raw.at };
  } catch {
    /* fall through */
  }
  return null;
}

export function clearPlanIntent(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* private mode */
  }
}

/** Account page link that opens this plan's checkout (see AccountView's ?upgrade=). */
export function upgradeHref(plan: PaidPlan, interval: BillingInterval): string {
  return `/account?upgrade=${plan}&interval=${interval}#plan`;
}

export const PLAN_NAME: Record<PaidPlan, string> = { pro: "Pro", premium: "Premium" };
