"use client";

import Link from "@/components/Link";
import { etDate } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

/** While the last charge for a paid plan failed: "update your card" on every page. */
export function BillingBanner() {
  const { me } = useSession();
  const due = me?.billing_attention;
  if (!me || !due || due.state !== "past_due") return null;
  const until = due.grace_ends_at ? etDate(due.grace_ends_at) : null;
  return (
    <aside className="install-hint billing-banner" role="status" aria-label="Payment problem">
      <p>
        <b>Your last payment didn&apos;t go through.</b> {until ? <>You keep {me.plan_label} until {until}. </> : null}
        Update your card to keep it.
      </p>
      <Link href="/account" className="btn btn-primary">Update payment</Link>
    </aside>
  );
}
