"use client";

// The link in every HSF email (/unsubscribe?t=…&k=…). Works signed out: the token names
// the account. Opening the page changes nothing (email scanners open links); a person
// has to press a button.
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ErrorLine, ErrorState, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";

type Kind = "digest" | "evening" | "alerts";
const LABELS: Record<Kind, string> = { digest: "Morning market digest", evening: "Evening market wrap", alerts: "Alert emails" };
const KINDS = Object.keys(LABELS) as Kind[];

export function UnsubscribeView() {
  const params = useSearchParams();
  const token = (params.get("t") || "").trim();
  const k = (params.get("k") || "").toLowerCase();
  const kind = (KINDS as string[]).includes(k) ? (k as Kind) : null;
  const valid = token.length >= 10 && token.length <= 64;
  const state = useApi(valid ? `unsub:${token}` : null, (signal) =>
    unwrap(publicApi.GET("/v1/email-preferences/unsubscribe", { params: { query: { t: token } }, signal })));
  const [saved, setSaved] = useState<Schemas["UnsubscribeState"] | null>(null);
  const act = useAction();
  const save = (which: Kind | "all") => void act.run(async () => {
    setSaved(await unwrap(publicApi.POST("/v1/email-preferences/unsubscribe", { body: { token, kind: which } })));
    return true;
  });
  const s = saved ?? state.data;
  const invalid = !valid || state.error?.status === 400;
  return (
    <div className="card narrow login">
      <p className="brand brand-lg">HSFinest<span>.AI</span></p>
      <h1 className="h1">Email preferences</h1>
      {invalid ? (
        <p className="form-error" role="alert">This unsubscribe link isn&apos;t valid. Sign in and open Account to change your emails.</p>
      ) : state.error && !s ? <ErrorState error={state.error} onRetry={state.reload} what="your email settings" />
        : !s ? <Skeleton rows={3} label="Loading your email settings" />
        : (
          <>
            <p className="body-sm">Emails for <span className="mono">{s.email}</span></p>
            {KINDS.every((x) => !s.prefs[x]) ? <p className="notice" role="status">You&apos;re unsubscribed from all HSF emails. In-app alerts still work.</p>
              : kind && !s.prefs[kind] ? <p className="notice" role="status">You&apos;re unsubscribed from the {LABELS[kind].toLowerCase()}.</p>
              : null}
            <ErrorLine error={act.error} />
            <div className="stack-sm">
              {kind && s.prefs[kind] && (
                <button type="button" className="btn btn-primary btn-block" disabled={act.busy} onClick={() => save(kind)}>
                  Unsubscribe from the {LABELS[kind].toLowerCase()}
                </button>
              )}
              {KINDS.some((x) => s.prefs[x]) && (
                <button type="button" className={`btn btn-block${kind && s.prefs[kind] ? "" : " btn-primary"}`} disabled={act.busy} onClick={() => save("all")}>
                  Unsubscribe from all HSF emails
                </button>
              )}
            </div>
            <ul className="bullets">{KINDS.map((x) => <li key={x}>{LABELS[x]}: {s.prefs[x] ? "on" : "off"}</li>)}</ul>
            <p className="cap">Changed your mind? Sign in and turn emails back on in Account. Account emails like password resets always go out.</p>
          </>
        )}
      <p className="cap login-links"><Link href="/account">Go to HSF</Link></p>
    </div>
  );
}
