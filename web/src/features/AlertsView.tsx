"use client";
import { tabKey } from "@/lib/tabs";

import Link from "@/components/Link";
import { useEffect, useId, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import { alerts, alertRules, emailPrefs } from "@/api/userData";
import { Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Skeleton, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { currentPushState, disablePush, enablePush } from "@/lib/webPush";
import type { PushState } from "@/lib/webPush";
import { groupFired } from "@/lib/alertEvents";
import { useSession } from "@/session/SessionProvider";

import { NotificationDelivery } from "./NotificationDelivery";
import { SmartAlerts } from "./SmartAlerts";
import { StarterAlerts } from "./Starters";
import { AlertComposer } from "./AlertComposer";
import { ActivityFilters, ActivityTable, DEFAULT_FILTER, FiredMost, TickersTable, inRange } from "./AlertActivity";
import type { ActivityFilter } from "./AlertActivity";
import { TYPES_UNAVAILABLE, typesUnavailable } from "./AlertForm";
import { AlertRulesTable } from "./AlertRulesTable";
import type { RuleFilter } from "./AlertRulesTable";

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

type AlertEvent = Awaited<ReturnType<typeof alerts.events>>[number];
const PAGE = 100;
type View = "activity" | "rules" | "tickers" | "delivery";
const VIEWS: [View, string][] = [["activity", "Activity"], ["rules", "Rules"], ["tickers", "Tickers"], ["delivery", "Delivery"]];

export function AlertsView() {
  const { me } = useSession();
  const list = useApi("alerts", (signal) => alerts.list(signal));
  const rules = useApi("alert-rules", (signal) => alertRules.list(signal));
  const types = useApi("alert-types", (signal) => alerts.types(signal));
  const events = useApi("alert-events", (signal) => alerts.events(signal, { limit: PAGE }));
  const [older, setOlder] = useState<AlertEvent[]>([]);
  const [olderMore, setOlderMore] = useState<boolean | null>(null);
  const olderAct = useAction();
  const [view, setView] = useState<View>("activity");
  const [chosen, setChosen] = useState<ActivityFilter | null>(null);
  const [ruleFilter, setRuleFilter] = useState<RuleFilter>("all");
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
  const allEvents = useMemo(() => {
    const seen = new Set<string>();
    return [...(events.data ?? []), ...older].filter((e) => (seen.has(e.event_id) ? false : (seen.add(e.event_id), true)));
  }, [events.data, older]);
  const items = useMemo(() => groupFired(allEvents), [allEvents]);
  const hasMore = olderMore ?? (events.data?.length ?? 0) >= PAGE;
  const loadOlder = () => {
    const cursor = allEvents[allEvents.length - 1]?.cursor;
    if (!cursor) return;
    void olderAct.run(async () => {
      const page = await alerts.events(undefined, { limit: PAGE, cursor });
      setOlder((o) => [...o, ...page]);
      setOlderMore(page.length >= PAGE);
      setFilter((f) => ({ ...f, range: "all" }));
      return true;
    });
  };
  const data = list.data;
  const full = !!data && data.used >= data.limit;
  const smart = rules.data?.rules ?? [];
  // The counts include HSF rules; until those load (or when they fail) they're unknown, not zero.
  const rulesKnown = !!rules.data || (!!rules.error && typesUnavailable(rules.error.status));
  const active = (data?.alerts.filter((a) => a.enabled).length ?? 0) + smart.filter((a) => a.enabled).length;
  const paused = (data?.alerts.filter((a) => !a.enabled).length ?? 0) + smart.filter((a) => !a.enabled).length;
  const firedWeek = items.filter((i) => inRange(i, "7d")).length;
  // Opens on the last 7 days, or on everything loaded when nothing fired that recently.
  const filter: ActivityFilter = chosen ?? (firedWeek === 0 ? { ...DEFAULT_FILTER, range: "all" } : DEFAULT_FILTER);
  const setFilter = (f: ActivityFilter | ((prev: ActivityFilter) => ActivityFilter)) => setChosen(typeof f === "function" ? f(filter) : f);
  const nextPlan = me?.plan === "basic" ? "pro" : me?.plan === "pro" ? "premium" : null;
  const ruleEmail = Array.isArray(rules.data?.capabilities?.delivery_channels) && rules.data!.capabilities.delivery_channels.includes("email");
  const go = (v: View) => { setView(v); document.getElementById(`alert-tab-${v}`)?.focus(); };
  const toRules = (f: RuleFilter) => { setRuleFilter(f); go("rules"); };
  const showTicker = (t: string) => { setFilter((f) => ({ ...f, ticker: t, kind: "", freshOnly: false })); go("activity"); };
  const chip = (on: boolean, onClick: () => void, title: string, children: ReactNode) => (
    <button type="button" className={`wl-chip${on ? " on" : ""}`} aria-pressed={on} title={title} onClick={onClick}>{children}</button>
  );
  return <div className="stack">
    <section className="page-head"><div><h1 className="h1">Alerts</h1><p className="cap">Scan-based alerts run after market scans; price alerts are checked about every minute.</p></div>
      <button type="button" className="btn btn-primary" onClick={() => setNewOpen(true)}>New alert</button>
    </section>
    {created && <p className="notice" role="status">Created: {created}.</p>}
    {list.error && !data ? <ErrorState error={list.error} onRetry={list.reload} what="your alerts" /> : !data ? <Skeleton rows={6} label="Loading alerts" /> : <>
      <div className="wl-summary al-summary" aria-label="Alert summary">
        {chip(view === "rules" && ruleFilter === "active", () => toRules("active"), "Show active alerts in Rules", <><strong>{rulesKnown ? active : "—"}</strong> active</>)}
        {chip(view === "rules" && ruleFilter === "paused", () => toRules("paused"), "Show paused alerts in Rules", <><strong>{rulesKnown ? paused : "—"}</strong> paused</>)}
        {chip(view === "rules" && ruleFilter === "legacy", () => toRules("legacy"), "Show price/technical alerts in Rules", <><strong>{data.used} / {data.limit}</strong> saved price/technical (including paused)</>)}
        {rules.data && chip(view === "rules" && ruleFilter === "all", () => toRules("all"), "Show every alert in Rules", <><strong>{rules.data.used} / {rules.data.limit}</strong> active alert allowance</>)}
        {chip(view === "delivery", () => go("delivery"), "Open Delivery", <><strong>{data.email_enabled ? "Available" : "Pro"}</strong> email delivery</>)}
        {events.data && firedWeek > 0 && chip(view === "activity" && filter.range === "7d" && !filter.kind && !filter.ticker && !filter.freshOnly,
          () => { setFilter({ ...DEFAULT_FILTER, range: "7d" }); go("activity"); }, "Show the last 7 days in Activity", <><strong>{firedWeek}</strong> fired in 7 days</>)}
      </div>
      <div className="wl-tabs" role="tablist" onKeyDown={tabKey} aria-label="Alert views">{VIEWS.map(([v, label]) =>
        <button type="button" role="tab" tabIndex={view === v ? 0 : -1} id={`alert-tab-${v}`} aria-controls={`alert-panel-${v}`} aria-selected={view === v} className={`wl-tab${view === v ? " on" : ""}`} key={v} onClick={() => setView(v)}>{label}</button>)}</div>
      <section id="alert-panel-activity" role="tabpanel" aria-labelledby="alert-tab-activity" hidden={view !== "activity"} className="stack">
        {data.alerts.length === 0 && smart.length === 0 && rules.data && <StarterAlerts rules={rules} onDone={(label) => { setCreated(label); reload(); }} />}
        <div className={items.length > 0 ? "al-split" : undefined}>
        <Card title="Recently fired" id="events">{events.error && !events.data ? <ErrorState error={events.error} onRetry={events.reload} what="recent alerts" /> : !events.data ? <Skeleton rows={3} label="Loading recent alerts" /> : events.data.length === 0 ? <Empty title="No recent triggers.">
          {active > 0 ? "Your alert is active; no trigger has been recorded yet."
            : paused > 0 ? "Your saved alerts are paused. View rules to manage them."
            : rules.error && !typesUnavailable(rules.error.status) ? "HSF rule status could not be loaded. View rules to retry."
            : !rules.data && !rules.error ? "Loading rule status…" : "Create an alert using New alert or from a stock page."}
          {(active > 0 || paused > 0 || !!rules.error && !typesUnavailable(rules.error.status)) && <div><button type="button" className="btn btn-sm" onClick={() => { setView("rules"); document.getElementById("alert-tab-rules")?.focus(); rules.reload(); }}>View rules</button></div>}
        </Empty> : <>
          <ActivityFilters items={items} filter={filter} onFilter={setFilter} />
          {filter.ticker && <p className="cap" role="status">Showing alerts with {filter.ticker}. <button type="button" className="link-btn" onClick={() => setFilter((f) => ({ ...f, ticker: "" }))}>Show all</button></p>}
          <ActivityTable items={items} filter={filter} onFilter={setFilter} next={hasMore} onLoadOlder={loadOlder} loadingOlder={olderAct.busy} olderError={<ErrorLine error={olderAct.error} />} />
        </>}</Card>
        {items.length > 0 && <FiredMost items={items} filter={filter} onPick={(t) => setFilter((f) => ({ ...f, ticker: t }))} onAll={() => go("tickers")} />}
        </div>
      </section>
      <section id="alert-panel-rules" role="tabpanel" aria-labelledby="alert-tab-rules" hidden={view !== "rules"} className="stack">
        <Card title="Your alerts" id="mine" aside={`${data.used} saved price/technical · ${smart.length} HSF rules`}>
          <p className="cap">Paused price/technical alerts still use saved slots. Pausing an HSF rule frees an active slot.</p>
          {data.alerts.length || smart.length ? <AlertRulesTable legacy={data.alerts} rules={smart} items={items} emailAllowed={ruleEmail}
            filter={ruleFilter} onFilter={setRuleFilter} onChanged={reload} />
            : (!rules.error || typesUnavailable(rules.error.status)) && !rules.loading ? <Empty title="No alerts yet.">Choose New alert to get started.</Empty> : null}
          {rules.error && !typesUnavailable(rules.error.status) && <ErrorState error={rules.error} onRetry={rules.reload} what="HSF alert rules" />}
          {data.used > data.limit && <p className="cap">Only your newest {data.limit} alerts can fire on this plan. Delete some or upgrade.</p>}
          {items.length > 0 && <p className="cap">“30 days” counts the alerts loaded on Activity.</p>}
        </Card>
        <SmartAlerts onChanged={reload} folded hideList rules={rules} />
      </section>
      <section id="alert-panel-tickers" role="tabpanel" aria-labelledby="alert-tab-tickers" hidden={view !== "tickers"} className="stack">
        {events.data && <TickersTable items={items} filter={filter} onFilter={setFilter} onShow={showTicker} />}
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
