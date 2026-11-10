"use client";

// Finishing the plan someone picked on Pricing before they had an account (lib/planIntent):
// - Account with ?upgrade=pro|premium opens that plan's checkout by itself once the email is
//   confirmed (the link from the verification page and the follow-up email);
// - Today shows a one-line reminder with the next step until they finish or dismiss it.
import Link from "@/components/Link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import { account } from "@/api/account";
import { ApiError } from "@/api/client";
import { useBillingInterval } from "@/components/BillingInterval";
import { beginBilling } from "@/lib/billingReturn";
import { checkout } from "@/lib/checkout";
import { PLAN_NAME, clearPlanIntent, parseInterval, parsePlan, readPlanIntent, upgradeHref } from "@/lib/planIntent";
import type { PaidPlan } from "@/lib/planIntent";
import { captureUsage } from "@/lib/usageAnalytics";
import type { Me } from "@/session/SessionProvider";

const RANK: Record<string, number> = { basic: 0, pro: 1, premium: 2, admin: 9 };

/** True when the account isn't on `plan` yet (a free trial of Pro doesn't count as Pro bought). */
export function needsPlan(me: Pick<Me, "plan" | "pro_trial_until">, plan: PaidPlan): boolean {
  const current = me.plan === "pro" && me.pro_trial_until ? "basic" : me.plan;
  return (RANK[current] ?? 0) < (RANK[plan] ?? 0);
}

/** Account page: opens checkout for ?upgrade= once the email is confirmed. */
export function AutoCheckout({ me }: { me: Me }) {
  const params = useSearchParams();
  const plan = parsePlan(params.get("upgrade"));
  const wanted = parseInterval(params.get("interval"));
  const [, setInterval] = useBillingInterval();
  const started = useRef(false);
  const [state, setState] = useState<"idle" | "opening" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const ready = !!plan && me.email_verified && me.plan !== "admin" && needsPlan(me, plan);
  useEffect(() => {
    if (!ready || started.current || !plan) return;
    started.current = true;
    setInterval(wanted);
    setState("opening");
    void (async () => {
      try {
        const link = await checkout(plan, wanted);
        captureUsage("checkout_started", { plan, interval: wanted });
        clearPlanIntent();
        beginBilling(me.plan, plan);
        window.location.assign(link.url);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Checkout couldn't open. Try the button below.");
        setState("error");
      }
    })();
  }, [ready, plan, wanted, me.plan, setInterval]);
  if (!plan || me.plan === "admin") return null;
  if (!needsPlan(me, plan)) return <p className="notice" role="status">You&apos;re already on {me.plan_label}.</p>;
  if (!me.email_verified) return <p className="notice" role="status">Confirm your email first, then {PLAN_NAME[plan]} checkout opens from the link. You can send a new link below.</p>;
  if (state === "opening") return <p className="notice" role="status">Opening {PLAN_NAME[plan]} checkout…</p>;
  if (state === "error") return <p className="form-error" role="alert">{error}</p>;
  return null;
}

const noSubscribe = () => () => {};
let snapshot: ReturnType<typeof readPlanIntent> | undefined;
const readOnce = () => (snapshot === undefined ? (snapshot = readPlanIntent()) : snapshot);

/** Today: "You picked Premium" with the next step, until it's done or dismissed. */
export function PlanIntentBanner({ me }: { me: Me }) {
  const intent = useSyncExternalStore(noSubscribe, readOnce, () => null);
  const [hidden, setHidden] = useState(false);
  const [sent, setSent] = useState<string | null>(null);
  if (!intent || hidden || me.plan === "admin" || !needsPlan(me, intent.plan)) return null;
  const name = PLAN_NAME[intent.plan];
  const dismiss = () => { clearPlanIntent(); snapshot = null; setHidden(true); };
  return (
    <div className="notice plan-intent" role="status">
      <span>
        {me.email_verified
          ? <>You picked {name}. <Link href={upgradeHref(intent.plan, intent.interval)} className="strong">Continue to checkout</Link></>
          : <>You picked {name}. Confirm your email from the link we sent, then checkout opens.{" "}
            <button type="button" className="link-btn" onClick={() => void account.resendVerification()
              .then((r) => setSent(r.message), () => setSent("Couldn't send it right now. Try again from Account."))}>Send a new link</button></>}
        {sent && <span className="cap"> {sent}</span>}
      </span>
      <button type="button" className="link-btn" onClick={dismiss} aria-label={`Dismiss the ${name} reminder`}>Not now</button>
    </div>
  );
}
