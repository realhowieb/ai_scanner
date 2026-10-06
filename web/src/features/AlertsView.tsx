"use client";

import Link from "next/link";
import { useState } from "react";

import { alerts, emailPrefs } from "@/api/userData";
import type { Alert } from "@/api/userData";
import { ConfirmDialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

import { AlertForm, TYPES_UNAVAILABLE, describeAlert, typesUnavailable } from "./AlertForm";

function AlertRow({ a, onChanged }: { a: Alert; onChanged: () => void }) {
  const toggle = useAction();
  const del = useAction();
  const [confirming, setConfirming] = useState(false);
  const label = describeAlert(a);
  return (
    <li className="alert-row">
      <div className="grow stack-xs">
        <span className="strong">{label}</span>
        <span className="cap">{a.last_fired_at ? `Last fired ${etTime(a.last_fired_at)}` : "Hasn't fired yet"}{a.created_at ? ` · Created ${etTime(a.created_at)}` : ""}</span>
        <ErrorLine error={toggle.error} />
      </div>
      <Pill tone={a.enabled ? "up" : undefined}>{a.enabled ? "On" : "Off"}</Pill>
      <button type="button" className="btn btn-sm" aria-pressed={a.enabled} disabled={toggle.busy}
        aria-label={`${a.enabled ? "Turn off" : "Turn on"}: ${label}`}
        onClick={() => void toggle.run(async () => { await alerts.setEnabled(a.id, !a.enabled); onChanged(); return true; })}>
        {toggle.busy ? "Saving…" : a.enabled ? "Turn off" : "Turn on"}
      </button>
      <button type="button" className="btn btn-sm btn-danger-outline" aria-label={`Delete: ${label}`} onClick={() => { del.clear(); setConfirming(true); }}>Delete</button>
      <ConfirmDialog open={confirming} title="Delete this alert?" confirmLabel="Delete alert" busy={del.busy}
        body={<p>{label}. Its history stays in recent alerts. This can&apos;t be undone.</p>} error={<ErrorLine error={del.error} />}
        onClose={() => setConfirming(false)}
        onConfirm={() => void del.run(async () => { await alerts.remove(a.id); setConfirming(false); onChanged(); return true; })} />
    </li>
  );
}

function EmailToggle() {
  const prefs = useApi("email-prefs", (signal) => emailPrefs.get(signal));
  const act = useAction();
  if (prefs.error) return <ErrorLine error={prefs.error} />;
  if (!prefs.data) return <p className="cap">Loading email settings…</p>;
  const on = prefs.data.alerts;
  return (
    <div className="stack-sm">
      <label className="check">
        <input type="checkbox" checked={on} disabled={act.busy}
          onChange={(e) => { const next = e.target.checked; void act.run(async () => { await emailPrefs.setAlerts(next); prefs.reload(); return true; }); }} />
        <span>Email me when an alert fires{act.busy ? " (saving…)" : ""}</span>
      </label>
      <ErrorLine error={act.error} />
    </div>
  );
}

export function AlertsView() {
  const { me } = useSession();
  const list = useApi("alerts", (signal) => alerts.list(signal));
  const types = useApi("alert-types", (signal) => alerts.types(signal));
  const events = useApi("alert-events", (signal) => alerts.events(signal));
  const [created, setCreated] = useState<string | null>(null);
  const [formKey, setFormKey] = useState(0);
  const reload = () => { list.reload(); events.reload(); };

  const data = list.data;
  const full = !!data && data.used >= data.limit;
  const nextPlan = me?.plan === "basic" ? "pro" : me?.plan === "pro" ? "premium" : null;

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Alerts</h1>
          <p className="cap">HSF checks your alerts automatically: scan-based ones a few times a day, live ones about every minute.</p>
        </div>
      </section>

      {list.error && !data ? <ErrorState error={list.error} onRetry={list.reload} what="your alerts" /> : !data ? <Skeleton rows={6} label="Loading alerts" /> : (
        <div className="split">
          <div className="col-main">
            <Card title="Your alerts" id="mine" aside={`Using ${data.used} of ${data.limit} on your ${me?.plan_label ?? ""} plan`}>
              <div className="track cap-track" aria-hidden="true"><div className="fill" style={{ width: `${Math.min(100, (data.used / Math.max(1, data.limit)) * 100)}%` }} /></div>
              {data.alerts.length === 0 ? (
                <Empty title="No alerts yet.">Create one here or from a stock page.</Empty>
              ) : (
                <ul className="alert-list">{data.alerts.map((a) => <AlertRow key={a.id} a={a} onChanged={reload} />)}</ul>
              )}
              {data.used > data.limit && <p className="cap">Only your newest {data.limit} alert{data.limit === 1 ? "" : "s"} can fire on this plan. Delete some or upgrade.</p>}
            </Card>

            <Card title="Recently fired" id="events">
              {events.error ? <ErrorLine error={events.error} /> : !events.data ? <Skeleton rows={3} label="Loading recent alerts" /> :
                events.data.length === 0 ? <Empty title="Nothing has fired yet.">When an alert&apos;s condition is met it shows here.</Empty> : (
                  <ul className="timeline">
                    {events.data.map((e) => <li key={e.id}><span className="cap">{etTime(e.fired_at)}</span> {e.ticker && <span className="mono strong">{e.ticker}</span>} {e.message}</li>)}
                  </ul>
                )}
            </Card>
          </div>

          <aside className="col-side">
            <Card title="New alert" id="new">
              {full ? (
                <div className="locked">
                  <p className="strong">You&apos;re using all {data.limit} alert{data.limit === 1 ? "" : "s"} on your plan.</p>
                  <p className="cap">Delete one to add another{nextPlan ? ", or upgrade for more" : ""}.</p>
                  {nextPlan && <UpgradeButton plan={nextPlan} />}
                </div>
              ) : types.error && typesUnavailable(types.error.status) ? <p className="notice" role="note">{TYPES_UNAVAILABLE}</p>
                : types.error ? <ErrorLine error={types.error} /> : !types.data ? <Skeleton rows={4} label="Loading alert types" /> : (
                <>
                  <AlertForm key={formKey} types={types.data} onCreated={(a) => { setCreated(describeAlert(a)); setFormKey((k) => k + 1); reload(); }} />
                  {created && <p className="notice" role="status">Created: {created}.</p>}
                </>
              )}
            </Card>

            <Card title="How you're notified" id="delivery">
              <p className="body-sm">Fired alerts appear under Recently fired here and in the classic app.</p>
              {data.email_enabled ? (
                <EmailToggle />
              ) : (
                <div className="stack-sm">
                  <p className="body-sm">Email alerts are part of Pro.</p>
                  <UpgradeButton plan="pro" />
                </div>
              )}
              <p className="cap">Phone push notifications aren&apos;t available yet.</p>
              <Link href="/watchlists" className="cap">Manage the watchlists your alerts use →</Link>
            </Card>
          </aside>
        </div>
      )}
    </div>
  );
}
