import { account } from "@/api/account";
import { ApiError } from "@/api/client";
import type { BillingInterval } from "@/components/BillingInterval";

/** Browser lock reduces repeat clicks across tabs; the API also locks/reuses the
 * durable checkout intent across browsers and API instances. */
export async function checkout(plan: "pro" | "premium", interval: BillingInterval) {
  if (typeof navigator !== "undefined" && navigator.locks) {
    return navigator.locks.request("hsf-checkout", { ifAvailable: true }, async (lock) => {
      if (!lock) throw new ApiError(409, "Checkout is opening in another window. Finish there or try again shortly.", null, null, null);
      return account.checkout(plan, interval);
    });
  }
  return account.checkout(plan, interval);
}
