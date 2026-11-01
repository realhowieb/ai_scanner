"use client";
import { tabKey } from "@/lib/tabs";

import Link from "@/components/Link";
import { useEffect, useId, useState } from "react";

import { api, unwrap } from "@/api/client";
import { alerts, alertRules, emailPrefs } from "@/api/userData";
import type { Alert } from "@/api/userData";
import { ConfirmDialog, Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { currentPushState, disablePush, enablePush } from "@/lib/webPush";
import type { PushState } from "@/lib/webPush";
import { groupFired } from "@/lib/alertEvents";
import type { FiredItem } from "@/lib/alertEvents";
import { etTime } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

import { NotificationDelivery } from "./NotificationDelivery";
import { SmartAlerts } from "./SmartAlerts";
import { StarterAlerts } from "./Starters";
import { AlertComposer } from "./AlertComposer";
import { TYPES_UNAVAILABLE, describeAlert, typesUnavailable } from "./AlertForm";

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
  const descriptionId = useId();
  const act = useAction();
  useEffect(() => {
    let live = true;
    const refresh = () => void currentPushState().then((s) => live && setState(s), () => live && setState("unsupported"));
    refresh();
    window.addEventListener("focus", refresh);
    return () => { live = false; window.removeEventListener("focus", refresh); };
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
        <input type="checkbox" checked={on} aria-describedby={descriptionId} disabled={act.busy || state === "unsupported" || state === "denied"}
          onChange={(e) => change(e.target.checked)} />
        <span>Notify me in this browser when an alert fires{act.busy ? " (saving…)" : ""}</span>
      </label>
      <p className="cap" id={descriptionId}>{PUSH_COPY[state] || "Uses your existing alert rules and quiet hours. Turn this on separately in each browser. Delivery depends on your browser and device settings."}</p>
      <ErrorLine error={act.error} />
    </div>
  );
}

export function AlertsView() {
  const { me } = useSession();
  const list = useApi("alerts", (signal) => alerts.list(signal));
  const rules = useApi("alert-rules", (signal) => alertRules.list(signal));
  const types = useApi("alert-types", (signal) => alerts.types(signal));
  const events = useApi("alert-events", (signal) => alerts.events(signal));
  const [view, setView] = useState<"activity" | "rules" | "delivery">("activity");
  const [newOpen, setNewOpen] = useState(false);
  const [created, setCreated] = useState<string | null>(null);
  const [formKey, setFormKey] = useState(0);
  const reload = () => { list.reload(); rules.reload(); events.reload(); };
  // Links from the Getting started card: /alerts?new=1 opens New alert, /alerts#delivery the Delivery tab.
  useEffect(() => {
    if (typeof window === "undefined") return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- applies the link's target once on arrival
    if (window.location.hash === "#delivery") setView("delivery");
    if (new URLSearchParams(window.location.search).get("new") === "1") setNewOpen(true);
  }, []);
  const data = list.data;
  const full = !!data && data.used >= data.limit;
  const smart = rules.data?.rules ?? [];
  // The counts include HSF rules; until those load (or when they fail) they're unknown, not zero.
  const rulesKnown = !!rules.data || (!!rules.error && typesUnavailable(rules.error.status));
  const active = (data?.alerts.filter((a) => a.enabled).length ?? 0) + smart.filter((a) => a.enabled).length;
  const paused = (data?.alerts.filter((a) => !a.enabled).length ?? 0) + smart.filter((a) => !a.enabled).length;
  const nextPlan = me?.plan === "basic" ? "pro" : me?.plan === "pro" ? "premium" : null;
  return <div className="stack">
    <section className="page-head"><div><h1 className="h1">Alerts</h1><p className="cap">Scan-based alerts run after market scans; price alerts are checked about every minute.</p></div>
      <button type="button" className="btn btn-primary" onClick={() => setNewOpen(true)}>New alert</button>
    </section>
    {created && <p className="notice" role="status">Created: {created}.</p>}
    {list.error && !data ? <ErrorState error={list.error} onRetry={list.reload} what="your alerts" /> : !data ? <Skeleton rows={6} label="Loading alerts" /> : <>
      <div className="metric-strip" aria-label="Alert summary">
        <span><strong>{rulesKnown ? active : "—"}</strong> active</span>
        <span><strong>{rulesKnown ? paused : "—"}</strong> paused</span>
        <span><strong>{data.used} / {data.limit}</strong> saved price/technical alerts (including paused)</span>
        {rules.data && <span><strong>{rules.data.used} / {rules.data.limit}</strong> active alert allowance</span>}
        <span><strong>{data.email_enabled ? "Available" : "Pro"}</strong> email delivery</span>
      </div>
      <div className="chips" role="tablist" onKeyDown={tabKey} aria-label="Alert views">{(["activity", "rules", "delivery"] as const).map((v) =>
        <button type="button" role="tab" tabIndex={view === v ? 0 : -1} id={`alert-tab-${v}`} aria-controls={`alert-panel-${v}`} aria-selected={view === v} className="chip" key={v} onClick={() => setView(v)}>{v[0]!.toUpperCase() + v.slice(1)}</button>)}</div>
      <section id="alert-panel-activity" role="tabpanel" aria-labelledby="alert-tab-activity" hidden={view !== "activity"} className="stack">
        {data.alerts.length === 0 && smart.length === 0 && rules.data && <StarterAlerts rules={rules} onDone={(label) => { setCreated(label); reload(); }} />}
        <Card title="Recently fired" id="events">{events.error && !events.data ? <ErrorState error={events.error} onRetry={events.reload} what="recent alerts" /> : !events.data ? <Skeleton rows={3} label="Loading recent alerts" /> : events.data.length === 0 ? <Empty title="No recent triggers.">
          {active > 0 ? "Your alert is active; no trigger has been recorded yet."
            : paused > 0 ? "Your saved alerts are paused. View rules to manage them."
            : rules.error && !typesUnavailable(rules.error.status) ? "HSF rule status could not be loaded. View rules to retry."
            : !rules.data && !rules.error ? "Loading rule status…" : "Create an alert using New alert or from a stock page."}
          {(active > 0 || paused > 0 || !!rules.error && !typesUnavailable(rules.error.status)) && <div><button type="button" className="btn btn-sm" onClick={() => { setView("rules"); document.getElementById("alert-tab-rules")?.focus(); rules.reload(); }}>View rules</button></div>}
        </Empty> : <ul className="fired-list">{groupFired(events.data).map((item) => <FiredRowItem key={item.key} item={item} />)}</ul>}</Card>
      </section>
      <section id="alert-panel-rules" role="tabpanel" aria-labelledby="alert-tab-rules" hidden={view !== "rules"} className="stack">
        <Card title="Your alerts" id="mine" aside={`${data.used} saved price/technical · ${smart.length} HSF rules`}>
          <p className="cap">Paused price/technical alerts still use saved slots. Pausing an HSF rule frees an active slot.</p>
          {data.alerts.length ? <ul className="alert-list">{data.alerts.map((a) => <AlertRow key={a.id} a={a} onChanged={reload} />)}</ul> : smart.length === 0 && (!rules.error || typesUnavailable(rules.error.status)) && !rules.loading ? <Empty title="No alerts yet.">Choose New alert to get started.</Empty> : null}
          {rules.error && !typesUnavailable(rules.error.status) && <ErrorState error={rules.error} onRetry={rules.reload} what="HSF alert rules" />}
          {data.used > data.limit && <p className="cap">Only your newest {data.limit} alerts can fire on this plan. Delete some or upgrade.</p>}
        </Card>
        <SmartAlerts onChanged={reload} folded rules={rules} />
      </section>
      <section id="alert-panel-delivery" role="tabpanel" aria-labelledby="alert-tab-delivery" hidden={view !== "delivery"} className="stack">
        <Card title="How you're notified" id="delivery"><p className="body-sm">Fired alerts appear in Activity.</p><PushToggle />
          {data.email_enabled ? <EmailToggle /> : <div className="stack-sm"><p className="body-sm">Email alerts are part of Pro.</p><UpgradeButton plan="pro" /></div>}
          <p className="cap">Browser notifications depend on this device’s permissions.</p><Link href="/watchlists">Manage alert watchlists →</Link>
        </Card>
        <NotificationDelivery events={events.data ?? []} />
      </section>
    </>}
    <Dialog open={newOpen} title="New alert" onClose={() => setNewOpen(false)}>
      {!data ? <Skeleton rows={3} label="Loading your alert allowance" /> : full && rules.error && typesUnavailable(rules.error.status) ? <div className="locked"><p className="strong">You&apos;re using all {data.limit} alert{data.limit === 1 ? "" : "s"} on your plan.</p><p className="cap">Delete one to add another{nextPlan ? ", or upgrade for more" : ""}.</p>{nextPlan && <UpgradeButton plan={nextPlan} />}</div>
        : types.error && typesUnavailable(types.error.status) ? <p className="notice" role="note">{TYPES_UNAVAILABLE}</p>
        : types.error ? <ErrorLine error={types.error} /> : !types.data ? <Skeleton rows={4} label="Loading alert types" />
        : <AlertComposer key={formKey} types={types.data} legacyFull={full} smartFull={!!rules.data && rules.data.used >= rules.data.limit} onCreated={(description) => { setCreated(description); setFormKey((k) => k + 1); setNewOpen(false); setView("rules"); reload(); }} />}
    </Dialog>
  </div>;
}
