"use client";

import Link from "@/components/Link";
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { captureUsage } from "@/lib/usageAnalytics";

type Data = Schemas["GettingStarted"];
type Step = Schemas["GettingStartedStep"];

/** Four first steps for a new account, at the top of Today. Ticked by the API from what the
 * account has done; hides itself once every step is done or the card is closed. */
export function GettingStarted() {
  const res = useApi("getting-started", (signal) => unwrap(api.GET("/v1/me/getting-started", { signal })));
  const [local, setLocal] = useState<Data | null>(null);
  const act = useAction();
  const data = local ?? res.data;
  if (!data || !data.show) return null;

  const dismiss = () => void act.run(async () => {
    const out = await unwrap(api.POST("/v1/me/getting-started/dismiss"));
    captureUsage("getting_started_dismissed", { done: data.done });
    setLocal(out);
  });
  // The guide is a public page, so opening it from here is what ticks the step.
  const open = (s: Step) => {
    captureUsage("getting_started_step_opened", { step: s.id });
    if (s.id === "guide" && !s.done) {
      void api.POST("/v1/me/getting-started/guide").then((r) => { if (r.data) setLocal(r.data); }).catch(() => undefined);
    }
  };
  const pct = Math.round((data.done / data.total) * 100);

  return (
    <Card title="Getting started" id="getting-started" className="getting-started"
      aside={<span className="mono">{data.done} of {data.total} done</span>}>
      <div className="gs-bar" role="progressbar" aria-label="Getting started progress" aria-valuemin={0}
        aria-valuemax={data.total} aria-valuenow={data.done}><span style={{ width: `${pct}%` }} /></div>
      <ol className="gs-steps">
        {data.steps.map((s) => (
          <li key={s.id} className={s.done ? "gs-done" : undefined}>
            <span className="gs-mark" aria-hidden="true">{s.done ? "✓" : ""}</span>
            <div className="grow">
              <p className="strong">{s.title}<span className="sr-only">{s.done ? " (done)" : " (to do)"}</span></p>
              {!s.done && <p className="cap">{s.detail}</p>}
            </div>
            {!s.done && <Link href={s.href} className="btn" onClick={() => open(s)}>{s.id === "guide" ? "Read" : "Start"}</Link>}
          </li>
        ))}
      </ol>
      <div className="card-foot">
        <span className="cap">Takes about five minutes. You can close this anytime.</span>
        <button type="button" className="link-btn cap" onClick={dismiss} disabled={act.busy}>Hide this</button>
      </div>
      <ErrorLine error={act.error} />
    </Card>
  );
}
