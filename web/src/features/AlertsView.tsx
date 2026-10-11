"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { api, unwrap } from "@/api/client";
import { alerts, emailPrefs } from "@/api/userData";
import type { Alert } from "@/api/userData";
import { ConfirmDialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { currentPushState, disablePush, enablePush } from "@/lib/webPush";
import type { PushState } from "@/lib/webPush";
import { groupFired } from "@/lib/alertEvents";
import type { FiredItem } from "@/lib/alertEvents";
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

const FIRED_SHOWN = 8;

function FiredRowItem({ item }: { item: FiredItem }) {
  const [open, setOpen] = useState(false);
  const rows = open ? item.rows : item.rows.slice(0, FIRED_SHOWN);
  const unlisted = item.total - item.rows.length;
  const many = item.rows.length > 1 || item.total > 1;
  const head = many ? `${item.title} · ${item.total} names` : item.ticker ? `${item.title} · ${item.ticker}` : item.title;
  return (
    <li className="fired">
      <span className="cap fired-time">{etTime(item.firedAt)}</span>
      <div className="stack-xs grow">
        <div className="fired-head">
          <span className="strong">{head}</span>
          {item.tiers.length > 1 && item.tiers.map((t) => <Pill key={t.rule}>{`≥ ${t.rule}: ${t.count}`}</Pill>)}
          {item.tiers.length === 1 && item.tiers[0]!.rule && <span className="cap">BreakoutScore ≥ {item.tiers[0]!.rule}</span>}
        </div>
        {rows.length > 0 && (
          <ul className="fired-chips" aria-label={`${item.title} matches`}>
            {rows.map((r) => (
              <li key={r.ticker} className="fired-chip" title={r.earnings ? `${r.detail} · ${r.earnings}` : r.detail}>
                <Link href={`/stocks/${encodeURIComponent(r.ticker)}`} className="mono strong">{r.ticker}</Link>
                {r.value !== null ? <span className="mono fired-val">{r.value.toFixed(1)}</span> : r.detail && <span className="cap">{r.detail}</span>}
                {r.earnings && <span className="fired-warn" aria-label={r.earnings}>⚠️</span>}
              </li>
            ))}
            {item.rows.length > FIRED_SHOWN && (
              <li><button type="button" className="link-btn cap" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
                {open ? "Show fewer" : `Show all ${item.rows.length}`}
              </button></li>
            )}
          </ul>
        )}
        {open && unlisted > 0 && <span className="cap">…and {unlisted} more not listed in the alert.</span>}
        {item.tiers.length > 1 && <span className="cap">Your ≥ {item.tiers.map((t) => t.rule).join(" and ≥ ")} breakout alerts fired on the same scan, shown together.</span>}
        {item.text && <span className="body-sm">{item.text}</span>}
      </div>
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

const PUSH_COPY: Record<PushState, string> = {
  unsupported: "This browser can't show notifications. On iPhone, add HSF to your Home Screen first.",
  denied: "Notifications are blocked for this site. Allow them in your browser's site settings, then come back.",
  off: "",
  on: "",
};

/** Browser notifications for every alert that fires (price alerts and alert rules). Hidden
 * until the server has its push keys; part of Pro. */
export function PushToggle() {
  const cfg = useApi("web-push-config", (signal) => unwrap(api.GET("/v1/web-push/config", { signal })));
  const [state, setState] = useState<PushState | null>(null);
  const act = useAction();
  useEffect(() => {
    let live = true;
    currentPushState().then((s) => live && setState(s), () => live && setState("unsupported"));
    return () => { live = false; };
  }, []);
  if (!cfg.data?.enabled || state === null) return null;
  if (cfg.data.allowed === false) {
    return state === "unsupported" ? null : (
      <div className="stack-sm">
        <p className="body-sm">Browser notifications are part of Pro.</p>
        <UpgradeButton plan="pro" />
      </div>
    );
  }
  if (!cfg.data.public_key) return null;
  const key = cfg.data.public_key;
  const on = state === "on";
  const change = (next: boolean) => void act.run(async () => {
    setState(next ? await enablePush(key) : (await disablePush(), "off"));
    return true;
  });
  return (
    <div className="stack-sm">
      <label className="check">
        <input type="checkbox" checked={on} disabled={act.busy || state === "unsupported"}
          onChange={(e) => change(e.target.checked)} />
        <span>Notify me in this browser when an alert fires{act.busy ? " (saving…)" : ""}</span>
      </label>
      {PUSH_COPY[state] && <p className="cap">{PUSH_COPY[state]}</p>}
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
                  <ul className="fired-list">
                    {groupFired(events.data).map((item) => <FiredRowItem key={item.key} item={item} />)}
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
              <PushToggle />
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
