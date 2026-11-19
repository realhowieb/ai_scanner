"use client";

// Alerts → Rules: price/technical alerts and HSF alert rules in one sortable table, with
// filters (active, paused, kind), on/off and email per row, and bulk pause/resume/email/delete.
import { useMemo, useState } from "react";

import { alertRules, alerts, watchlists } from "@/api/userData";
import type { Alert, AlertRule } from "@/api/userData";
import { ConfirmDialog } from "@/components/Dialog";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import type { FiredItem } from "@/lib/alertEvents";
import { etTime } from "@/lib/format";

import { KindTag } from "./AlertActivity";
import { describeAlert } from "./AlertForm";
import { describeRule } from "./SmartAlerts";

export type RuleFilter = "all" | "active" | "paused" | "legacy" | "hsf";

type Row = {
  key: string; source: "alert" | "rule"; id: number; label: string; appliesTo: string; kind: string;
  enabled: boolean; lastFired: string | null; sub: string; email: boolean | null; fired: number;
};

const LEGACY_KIND: Record<string, string> = { breakout: "Breakout", watchlist: "Watchlist", price: "Price", move: "Move", rvol: "RVOL", trigger: "Trigger", ema_cross: "Technical", ewo_cross: "Technical" };
function ruleKind(t: string): string {
  if (t === "SCAN_MATCH") return "Scan";
  if (t === "RVOL_ABOVE") return "RVOL";
  if (t === "EARNINGS_SOON") return "Earnings";
  if (t === "PREBREAKOUT_ACTIVE") return "Breakout";
  return "HSF";
}

const DAY_MS = 24 * 60 * 60 * 1000;

type SortKey = "label" | "kind" | "last" | "fired";
const SORT_LABEL: Record<SortKey, string> = { label: "Rule", kind: "Kind", last: "Last fired", fired: "Fired in 30 days" };

export function AlertRulesTable({ legacy, rules, items, emailAllowed, filter, onFilter, onChanged }: {
  legacy: Alert[]; rules: AlertRule[]; items: FiredItem[]; emailAllowed: boolean;
  filter: RuleFilter; onFilter: (f: RuleFilter) => void; onChanged: () => void;
}) {
  const types = useApi("alert-rule-types", (signal) => alertRules.types(signal));
  const lists = useApi("watchlists", (signal) => watchlists.list(signal));
  const [sel, setSel] = useState<string[]>([]);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "last", desc: true });
  const [confirm, setConfirm] = useState<Row[] | null>(null);
  const act = useAction();
  const bulk = useAction();

  const fired = useMemo(() => {
    const now = Date.now();
    const m = new Map<string, number>();
    for (const it of items) {
      const t = it.firedAt ? new Date(it.firedAt).getTime() : NaN;
      if (Number.isNaN(t) || now - t > 30 * DAY_MS) continue;
      for (const s of it.sources) m.set(s, (m.get(s) ?? 0) + 1);
    }
    return m;
  }, [items]);

  const all: Row[] = useMemo(() => {
    const listName = (id: number | null | undefined) => lists.data?.find((w) => w.id === id)?.name ?? "a watchlist";
    const ruleTypes = Array.isArray(types.data) ? types.data : [];
    return [
      ...legacy.map((a): Row => ({
        key: `alert:${a.id}`, source: "alert", id: a.id, label: describeAlert(a),
        appliesTo: a.ticker ?? (a.type === "watchlist" || a.watchlist_only ? "Watchlist tickers" : "All stocks"),
        kind: LEGACY_KIND[a.type] ?? "Other", enabled: a.enabled, lastFired: a.last_fired_at ?? null,
        sub: `${a.last_fired_at ? `Last fired ${etTime(a.last_fired_at)}` : "Hasn't fired yet"}${a.created_at ? ` · Created ${etTime(a.created_at)}` : ""}`,
        email: null, fired: fired.get(`alert:${a.id}`) ?? 0,
      })),
      ...rules.map((r): Row => {
        const scope = r.rule_type === "SCAN_MATCH" ? "Scan alert" : r.ticker ?? `Every stock on ${listName(r.watchlist_id)}`;
        return {
          key: `rule:${r.id}`, source: "rule", id: r.id, label: `${scope}: ${describeRule(r, ruleTypes)}`,
          appliesTo: r.rule_type === "SCAN_MATCH" ? "Scanner matches" : r.ticker ?? listName(r.watchlist_id),
          kind: ruleKind(r.rule_type), enabled: r.enabled, lastFired: r.last_triggered_at ?? null,
          sub: `${r.last_evaluated_at ? `Last evaluated ${etTime(r.last_evaluated_at)}` : "No evaluation recorded yet"} · ${r.last_triggered_at ? `Last fired ${etTime(r.last_triggered_at)}` : "Hasn't fired yet"}`,
          email: r.delivery_channels.includes("email"), fired: fired.get(`rule:${r.id}`) ?? 0,
        };
      }),
    ];
  }, [legacy, rules, types.data, lists.data, fired]);

  const counts = {
    all: all.length, active: all.filter((r) => r.enabled).length, paused: all.filter((r) => !r.enabled).length,
    legacy: legacy.length, hsf: rules.length,
  };
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    const keep = all.filter((r) => (filter === "all" || (filter === "active" && r.enabled) || (filter === "paused" && !r.enabled)
      || (filter === "legacy" && r.source === "alert") || (filter === "hsf" && r.source === "rule"))
      && (!t || `${r.label} ${r.appliesTo} ${r.kind}`.toLowerCase().includes(t)));
    const val = (r: Row): number | string => sort.key === "label" ? r.label.toLowerCase() : sort.key === "kind" ? r.kind
      : sort.key === "fired" ? r.fired : r.lastFired ? new Date(r.lastFired).getTime() : 0;
    return [...keep].sort((a, b) => {
      const x = val(a), y = val(b);
      const c = typeof x === "string" ? x.localeCompare(String(y)) : x - (y as number);
      return sort.desc ? -c : c;
    });
  }, [all, filter, q, sort]);

  const selected = all.filter((r) => sel.includes(r.key));
  const shownKeys = rows.map((r) => r.key);
  const allShown = rows.length > 0 && shownKeys.every((k) => sel.includes(k));
  const toggle = (k: string) => setSel((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]));

  const setEnabled = (r: Row, on: boolean) => (r.source === "alert" ? alerts.setEnabled(r.id, on) : alertRules.update(r.id, { enabled: on }));
  const setEmail = (r: Row, on: boolean) => alertRules.update(r.id, { delivery_channels: on ? ["in_app", "email"] : ["in_app"] });
  const remove = (r: Row) => (r.source === "alert" ? alerts.remove(r.id) : alertRules.remove(r.id));
  const runBulk = (fn: (r: Row) => Promise<unknown>, which: Row[]) => void bulk.run(async () => {
    for (const r of which) await fn(r);
    onChanged();
    return true;
  });

  const th = (key: SortKey, cls = "") => (
    <th scope="col" className={cls} aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      <button type="button" className="th-sort" aria-label={`Sort by ${SORT_LABEL[key]}`}
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key === "last" || key === "fired" }))}>
        {key === "fired" ? "30 days" : SORT_LABEL[key]}{sort.key === key && <span aria-hidden="true" className="th-arrow">{sort.desc ? " ▼" : " ▲"}</span>}
      </button>
    </th>
  );
  const FILTERS: [RuleFilter, string][] = [["all", "All"], ["active", "Active"], ["paused", "Paused"], ["legacy", "Price/technical"], ["hsf", "HSF rules"]];

  if (all.length === 0) return null;
  return (
    <div className="stack-sm">
      <div className="al-filters">
        <div className="seg-group" role="radiogroup" aria-label="Show rules">
          {FILTERS.filter(([f]) => f === "all" || counts[f] > 0 || filter === f).map(([f, label]) => (
            <label key={f} className={`seg${filter === f ? " on" : ""}`}>
              <input type="radio" className="sr-only" name="al-rule-filter" checked={filter === f} onChange={() => onFilter(f)} />{label} {counts[f]}
            </label>
          ))}
        </div>
        <label className="field al-search"><span className="sr-only">Filter rules</span>
          <input type="search" placeholder="Filter rules…" value={q} aria-label="Filter rules" onChange={(e) => setQ(e.target.value)} />
        </label>
      </div>
      {selected.length > 0 && (
        <div className="wl-bulk" role="region" aria-label="Selected alerts">
          <strong>{selected.length} selected</strong>
          <span className="grow" />
          <button type="button" className="btn btn-sm" disabled={bulk.busy} onClick={() => runBulk((r) => setEnabled(r, false), selected.filter((r) => r.enabled))}>Pause</button>
          <button type="button" className="btn btn-sm" disabled={bulk.busy} onClick={() => runBulk((r) => setEnabled(r, true), selected.filter((r) => !r.enabled))}>Resume</button>
          {emailAllowed && selected.some((r) => r.source === "rule") && <>
            <button type="button" className="btn btn-sm" disabled={bulk.busy} onClick={() => runBulk((r) => setEmail(r, true), selected.filter((r) => r.email === false))}>Email on</button>
            <button type="button" className="btn btn-sm" disabled={bulk.busy} onClick={() => runBulk((r) => setEmail(r, false), selected.filter((r) => r.email === true))}>Email off</button>
          </>}
          <button type="button" className="btn btn-sm btn-danger-outline" disabled={bulk.busy} onClick={() => { bulk.clear(); setConfirm(selected); }}>Delete selected</button>
          <button type="button" className="link-btn" onClick={() => setSel([])}>Clear</button>
        </div>
      )}
      <ErrorLine error={bulk.error ?? act.error} />
      {rows.length === 0 ? <p className="cap" role="status">No alerts match. <button type="button" className="link-btn" onClick={() => { onFilter("all"); setQ(""); }}>Show all</button></p> : (
        <div className="table-wrap" role="region" aria-label="Your alert rules" tabIndex={0}>
          <table className="table al-table al-rules">
            <thead>
              <tr>
                <th scope="col" className="wl-check"><input type="checkbox" aria-label="Select all shown alerts" checked={allShown}
                  onChange={() => setSel((s) => (allShown ? s.filter((k) => !shownKeys.includes(k)) : [...new Set([...s, ...shownKeys])]))} /></th>
                {th("label")}
                <th scope="col" className="al-mid">Applies to</th>
                {th("kind", "al-mid")}
                {th("last", "num")}
                {th("fired", "num al-mid")}
                {emailAllowed && <th scope="col" className="al-mid">Email</th>}
                <th scope="col">On</th>
                <th scope="col"><span className="sr-only">Delete</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} className={`${sel.includes(r.key) ? "sel" : ""}${r.enabled ? "" : " off"}`}>
                  <td className="wl-check"><input type="checkbox" aria-label={`Select ${r.label}`} checked={sel.includes(r.key)} onChange={() => toggle(r.key)} /></td>
                  <td className="al-rule"><span className="strong">{r.label}</span><span className="cap al-sub">{r.sub}</span></td>
                  <td className="al-mid">{r.appliesTo}</td>
                  <td className="al-mid"><KindTag title={r.kind} /></td>
                  <td className="num cap">{r.lastFired ? etTime(r.lastFired) : "Hasn't fired"}</td>
                  <td className="num mono al-mid">{r.fired}</td>
                  {emailAllowed && (
                    <td className="al-mid">{r.email === null ? <span className="cap" title="Price/technical alerts follow your email setting in Delivery" aria-label="Follows your email setting">–</span> : (
                      <button type="button" className="link-btn" disabled={act.busy} aria-pressed={r.email}
                        aria-label={`${r.email ? "Stop emailing" : "Email me"}: ${r.label}`}
                        onClick={() => void act.run(async () => { await setEmail(r, !r.email); onChanged(); return true; })}>{r.email ? "On" : "Off"}</button>
                    )}</td>
                  )}
                  <td>
                    <button type="button" className={`al-switch${r.enabled ? " on" : ""}`} aria-pressed={r.enabled} disabled={act.busy}
                      aria-label={`${r.enabled ? "Turn off" : "Turn on"}: ${r.label}`}
                      onClick={() => void act.run(async () => { await setEnabled(r, !r.enabled); onChanged(); return true; })}>
                      <span className="sr-only">{r.enabled ? "On" : "Off"}</span>
                    </button>
                  </td>
                  <td className="num"><button type="button" className="al-del" aria-label={`Delete: ${r.label}`} title="Delete" onClick={() => { bulk.clear(); setConfirm([r]); }}>×</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <ConfirmDialog open={!!confirm} title={confirm && confirm.length > 1 ? `Delete ${confirm.length} alerts?` : "Delete this alert?"}
        confirmLabel={confirm && confirm.length > 1 ? `Delete ${confirm.length} alerts` : "Delete alert"} busy={bulk.busy}
        body={<p>{confirm && confirm.length === 1 ? `${confirm[0]!.label}. ` : ""}Past alerts stay in Activity. This can&apos;t be undone.</p>}
        error={<ErrorLine error={bulk.error} />}
        onClose={() => setConfirm(null)}
        onConfirm={() => void bulk.run(async () => {
          for (const r of confirm ?? []) await remove(r);
          setSel((s) => s.filter((k) => !(confirm ?? []).some((r) => r.key === k)));
          setConfirm(null);
          onChanged();
          return true;
        })} />
    </div>
  );
}
