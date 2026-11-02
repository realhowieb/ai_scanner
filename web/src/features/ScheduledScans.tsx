"use client";

// Premium: save the Custom scan form as a scan that runs by itself on every trading day
// (before the open, midday or after the close) and emails the top results.
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ConfirmDialog } from "@/components/Dialog";
import { Card, ErrorLine, Locked, Pill } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import type { ScanCreate } from "@/hooks/useScanJob";
import { captureUsage } from "@/lib/usageAnalytics";
import { useSession } from "@/session/SessionProvider";

type Scheduled = Schemas["ScheduledScan"];
type RunAt = Scheduled["run_at"];

export const RUN_AT: { id: RunAt; label: string; when: string }[] = [
  { id: "open", label: "Before the open", when: "about 8:05 to 9:25 a.m. ET" },
  { id: "midday", label: "Midday", when: "about noon to 1:30 p.m. ET" },
  { id: "close", label: "After the close", when: "about 4:20 to 6 p.m. ET" },
];
export const MAX_SCHEDULED = 3;

const UNIVERSE_TEXT: Record<string, string> = {
  sp500: "S&P 500", nasdaq: "NASDAQ", combo: "Combo", us_market: "US market", watchlist: "a watchlist", ticker: "one ticker", earnings: "earnings reporters",
};

export function describeSchedule(s: Pick<Scheduled, "run_at" | "request">): string {
  const r = s.request as Partial<ScanCreate>;
  const f = (r.filters ?? {}) as NonNullable<ScanCreate["filters"]>;
  const where = r.universe === "ticker" && r.ticker ? r.ticker : UNIVERSE_TEXT[String(r.universe)] ?? String(r.universe ?? "");
  const when = RUN_AT.find((x) => x.id === s.run_at)?.label.toLowerCase() ?? s.run_at;
  const session = f.session && f.session !== "regular" ? ` · ${f.session === "premarket" ? "pre-market" : "after-hours"}` : "";
  return `${where}${session} · top ${f.top_n ?? 25} · every trading day ${when}`;
}

export function statusText(s: Pick<Scheduled, "last_status" | "last_detail" | "last_run_at">): { text: string; tone?: "warn" | "up" } {
  const at = s.last_run_at ? new Date(s.last_run_at) : null;
  const day = at && !Number.isNaN(at.getTime()) ? at.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" }) : "";
  switch (s.last_status) {
    case null: case undefined: case "": return { text: "Waiting for its first run" };
    case "running": return { text: "Running now" };
    case "sent": return { text: `Emailed ${day}${s.last_detail ? ` · ${s.last_detail}` : ""}`, tone: "up" };
    case "unsubscribed": return { text: "Ran, but alert emails are off in Account settings", tone: "warn" };
    case "unverified": return { text: "Ran, but your email isn't verified yet", tone: "warn" };
    case "paused": return { text: s.last_detail || "Paused", tone: "warn" };
    case "failed": return { text: s.last_detail || "The last run didn't finish", tone: "warn" };
    default: return { text: s.last_detail || "The email didn't send", tone: "warn" };
  }
}

export function ScheduledScans({ current }: { current: () => { req: ScanCreate } | { error: string } }) {
  const { can } = useSession();
  const allowed = can("can_scheduled_scans");
  const list = useApi(allowed ? "scheduled-scans" : null, (signal) => unwrap(api.GET("/v1/scheduled-scans", { signal })));
  const [name, setName] = useState("");
  const [runAt, setRunAt] = useState<RunAt>("open");
  const [formErr, setFormErr] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<Scheduled | null>(null);
  const save = useAction();
  const del = useAction();

  if (!allowed) {
    return (
      <Card title="Scheduled scans" id="scheduled-scans">
        <Locked title="Scheduled scans are part of Premium" plan="premium">
          Save this scan once and HSF runs it every trading day, before the open, at midday or after the close, then emails you the top results.
        </Locked>
      </Card>
    );
  }

  const rows = list.data ?? [];
  const full = rows.length >= MAX_SCHEDULED;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    setFormErr(null);
    const built = current();
    if ("error" in built) return setFormErr(built.error);
    const label = name.trim();
    if (!label) return setFormErr("Give this scan a name, like \"Morning gappers\".");
    void save.run(async () => {
      await unwrap(api.POST("/v1/scheduled-scans", { body: { name: label, run_at: runAt, scan: built.req as Schemas["ScheduledScanCreate"]["scan"] } }));
      captureUsage("scheduled_scan_saved", { universe: built.req.universe });
      setName("");
      list.reload();
      return true;
    });
  };

  return (
    <Card title="Scheduled scans" id="scheduled-scans" aside={`${rows.length} of ${MAX_SCHEDULED}`}>
      <p className="cap">Save the scan above and HSF runs it on every trading day, then emails you the top 10 results ranked by HSF Score. The results also show in your scan history.</p>
      {rows.length > 0 && (
        <ul className="sched-list">
          {rows.map((s) => {
            const st = statusText(s);
            return (
              <li key={s.id} className="sched-item">
                <div className="stack-xs">
                  <span className="strong">{s.name}</span>
                  <span className="cap">{describeSchedule(s)}</span>
                  <span>{st.tone ? <Pill tone={st.tone}>{st.text}</Pill> : <span className="cap">{st.text}</span>}</span>
                </div>
                <button type="button" className="btn btn-sm" onClick={() => { del.clear(); setDeleting(s); }} aria-label={`Delete scheduled scan ${s.name}`}>Delete</button>
              </li>
            );
          })}
        </ul>
      )}
      <ErrorLine error={list.error} />
      {full ? (
        <p className="cap">You have the most scheduled scans allowed. Delete one to add another.</p>
      ) : (
        <form className="sched-form" onSubmit={submit} aria-label="Schedule this scan">
          <label className="field">
            <span>Name</span>
            <input type="text" value={name} maxLength={60} placeholder="Morning gappers" onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field">
            <span>When</span>
            <select value={runAt} onChange={(e) => setRunAt(e.target.value as RunAt)}>
              {RUN_AT.map((r) => <option key={r.id} value={r.id}>{r.label} ({r.when})</option>)}
            </select>
          </label>
          <button type="submit" className="btn" disabled={save.busy}>{save.busy ? "Saving…" : "Email me this scan every trading day"}</button>
          {formErr && <p className="form-error" role="alert">{formErr}</p>}
          <ErrorLine error={save.error} />
        </form>
      )}
      <ConfirmDialog open={deleting !== null} title={`Delete "${deleting?.name ?? ""}"?`} confirmLabel="Delete" busy={del.busy}
        body={<p>This scan stops running and emailing you. Your past results stay in scan history.</p>}
        error={<ErrorLine error={del.error} />} onClose={() => setDeleting(null)}
        onConfirm={() => void del.run(async () => {
          await unwrap(api.DELETE("/v1/scheduled-scans/{schedule_id}", { params: { path: { schedule_id: deleting!.id } } }));
          setDeleting(null);
          list.reload();
          return true;
        })} />
    </Card>
  );
}
