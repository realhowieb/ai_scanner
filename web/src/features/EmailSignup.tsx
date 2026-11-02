"use client";

// "Get the weekly setups" email list form for the public pages, and the page behind the
// confirm / unsubscribe link (/subscribe/confirm?t=…). Double opt-in: the API emails a
// confirm link and the address joins only after it is opened. The answer is the same
// whether or not the address was already on the list. Opening the link page changes
// nothing (mail scanners open links); a person presses the button.
import Link from "@/components/Link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import type { FormEvent } from "react";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ErrorLine, ErrorState, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { captureAttribution } from "@/lib/funnel";

export const LIST_PITCH = "Once a week: the setups HSF's scanner ranked most and the biggest earnings coming up. Free, no account needed, unsubscribe any time.";

export function EmailSignup({ surface, title = "Get the weekly setups by email" }: { surface: string; title?: string }) {
  const config = useApi("list-config", (signal) => unwrap(publicApi.GET("/v1/public/list", { signal })));
  const [email, setEmail] = useState("");
  const [trap, setTrap] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const act = useAction();
  if (config.data && !config.data.enabled) return null;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    void act.run(async () => {
      const r = await unwrap(publicApi.POST("/v1/public/subscribe", {
        body: { email: email.trim(), surface, attribution: captureAttribution() as Record<string, string>, website: trap || null },
      }));
      setDone(r.message);
      return true;
    });
  };
  return (
    <section className="card stack-sm email-signup" aria-labelledby={`list-${surface}`}>
      <h2 id={`list-${surface}`} className="h3">{title}</h2>
      <p className="body-sm">{LIST_PITCH}</p>
      {done ? <p className="notice" role="status">{done}</p> : (
        <form className="inline-form" onSubmit={submit}>
          <label className="field grow"><span className="sr-only">Email address</span>
            <input type="email" required autoComplete="email" inputMode="email" placeholder="you@example.com" maxLength={254}
              value={email} onChange={(e) => setEmail(e.target.value)} />
          </label>
          {/* People never see this field; bots fill it in. */}
          <input type="text" name="website" tabIndex={-1} autoComplete="off" aria-hidden="true" className="hp-field"
            value={trap} onChange={(e) => setTrap(e.target.value)} />
          <button type="submit" className="btn btn-primary" disabled={act.busy}>Send me the weekly setups</button>
        </form>
      )}
      <ErrorLine error={act.error} />
      <p className="cap">Research only, not financial advice. We&apos;ll email you a link to confirm first.</p>
    </section>
  );
}

export function ListConfirmView() {
  const params = useSearchParams();
  const token = (params.get("t") || "").trim();
  const wantsOut = params.get("action") === "unsubscribe";
  const valid = token.length >= 10 && token.length <= 64;
  const state = useApi(valid ? `list:${token}` : null, (signal) =>
    unwrap(publicApi.GET("/v1/public/subscribe/status", { params: { query: { t: token } }, signal })));
  const [saved, setSaved] = useState<Schemas["ListStatus"] | null>(null);
  const act = useAction();
  const go = (action: "confirm" | "unsubscribe") => void act.run(async () => {
    setSaved(await unwrap(publicApi.POST("/v1/public/subscribe/action", { body: { token, action } })));
    return true;
  });
  const s = saved ?? state.data;
  const invalid = !valid || state.error?.status === 400;
  return (
    <div className="card narrow login">
      <Link href="/" className="brand brand-lg" aria-label="HSFinest.com home">HSFinest<span>.com</span></Link>
      <h1 className="h1">Weekly setups email</h1>
      {invalid ? (
        <p className="form-error" role="alert">This link isn&apos;t valid. Sign up again from the HSF home page.</p>
      ) : state.error && !s ? <ErrorState error={state.error} onRetry={state.reload} what="your email list status" />
        : !s ? <Skeleton rows={2} label="Loading" />
        : (
          <>
            <p className="body-sm">For <span className="mono">{s.email}</span></p>
            {s.status === "confirmed" && <p className="notice" role="status">You&apos;re on the list. The weekly setups email comes on weekends.</p>}
            {s.status === "unsubscribed" && <p className="notice" role="status">You&apos;re unsubscribed. You won&apos;t get the weekly setups email.</p>}
            <ErrorLine error={act.error} />
            <div className="stack-sm">
              {s.status === "pending" && !wantsOut && (
                <button type="button" className="btn btn-primary btn-block" disabled={act.busy} onClick={() => go("confirm")}>Confirm my email</button>
              )}
              {s.status !== "unsubscribed" && (
                <button type="button" className={`btn btn-block${wantsOut ? " btn-primary" : ""}`} disabled={act.busy} onClick={() => go("unsubscribe")}>
                  Unsubscribe
                </button>
              )}
            </div>
          </>
        )}
      <p className="cap login-links"><Link href="/weekly">This week&apos;s recap</Link> · <Link href="/signup">Create a free account</Link></p>
    </div>
  );
}
