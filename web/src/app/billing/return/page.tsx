import Link from "@/components/Link";

import { SignedInAs } from "./SignedInAs";

export const metadata = { title: "Return from billing" };

export default function BillingReturn() {
  return <main className="center"><div className="card narrow login stack">
    <p className="brand brand-lg">HSFinest<span>.AI</span></p>
    <h1 className="h1">Return to HSF</h1>
    <p>Your billing session has finished. HSF will check your subscription with Stripe before updating your plan.</p>
    <p>If you use HSF as a Mac web app, switch back to its window. Your plan refreshes automatically when you return.</p>
    <SignedInAs />
    <p>To continue in this browser, open your account. You may need to sign in.</p>
    <div><Link className="btn btn-primary" href="/account?billing=return">Open account and refresh plan</Link></div>
  </div></main>;
}
