"use client";

// Smart alerts: alert rules on HSF intelligence (score crossings, new setups, volume,
// rank moves, earnings heads-ups), evaluated by the server after each market scan.
// One card lists every rule with on/off, email and delete; its form adds a rule for a
// ticker or a whole watchlist. On a stock page the card is locked to that ticker.
// An API without the rule routes shows no card.
import Link from "next/link";
import { useState } from "react";
import type { FormEvent } from "react";

import { alertRules, watchlists } from "@/api/userData";
import type { AlertRule, AlertRuleType } from "@/api/userData";
import { Card, ErrorLine, Pill, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";

import { describeScanAlert } from "./ScanAlert";

/** The unit after a rule's number, so "1" reads as a day and "2" as a volume multiple. */
export function thresholdUnit(type: string): string {
  if (type === "EARNINGS_SOON") return "days";
  if (type === "RVOL_ABOVE") return "x volume";
  if (type === "RANK_IMPROVED") return "places";
  return "";
}

const PLAN_OF: Record<string, "pro" | "premium"> = { EARNINGS_SOON: "pro", SCAN_MATCH: "pro", PREBREAKOUT_ACTIVE: "premium" };

export function describeRule(r: Pick<AlertRule, "rule_type" | "threshold" | "value">, types: AlertRuleType[]): string {
  if (r.rule_type === "SCAN_MATCH") return `New scanner matches (${describeScanAlert(r.threshold, r.value)})`;
  const t = types.find((x) => x.type === r.rule_type);
  const base = (t?.label ?? r.rule_type).replace(/ \(days\)$/, "");
  const unit = thresholdUnit(r.rule_type);
  const num = r.threshold != null ? ` ${r.threshold}${unit ? ` ${unit}` : ""}` : "";
  return `${base}${num}${r.value ? `: ${r.value}` : ""}`;
}

function RuleRow({ r, types, scope, emailAllowed, onChanged }: {
  r: AlertRule; types: AlertRuleType[]; scope: string; emailAllowed: boolean; onChanged: () => void;
}) {
  const act = useAction();
  const label = `${scope}: ${describeRule(r, types)}`;
  const email = r.delivery_channels.includes("email");
  const change = (body: Parameters<typeof alertRules.update>[1]) =>
    void act.run(async () => { await alertRules.update(r.id, body); onChanged(); return true; });
  return (
    <li className="alert-row">
      <div className="grow stack-xs">
        <span className="strong">{label}</span>
        <span className="cap">
          {r.last_evaluated_at ? `Last evaluated ${etTime(r.last_evaluated_at)}` : "No evaluation recorded yet"} · {r.last_triggered_at ? `Last fired ${etTime(r.last_triggered_at)}` : "Hasn't fired yet"}
          {email ? " · Email on" : ""}
        </span>
        <ErrorLine error={act.error} />
      </div>
      <Pill tone={r.enabled ? "up" : undefined}>{r.enabled ? "On" : "Off"}</Pill>
      <button type="button" className="btn btn-sm" disabled={act.busy} aria-pressed={r.enabled}
        aria-label={`${r.enabled ? "Turn off" : "Turn on"}: ${label}`} onClick={() => change({ enabled: !r.enabled })}>
        {r.enabled ? "Turn off" : "Turn on"}
      </button>
      {emailAllowed && (
        <button type="button" className="btn btn-sm" disabled={act.busy} aria-pressed={email}
          aria-label={`${email ? "Stop emailing" : "Email me"}: ${label}`}
          onClick={() => change({ delivery_channels: email ? ["in_app"] : ["in_app", "email"] })}>
          {email ? "No email" : "Email me"}
        </button>
      )}
      <button type="button" className="btn btn-sm btn-danger-outline" disabled={act.busy} aria-label={`Delete: ${label}`}
        onClick={() => void act.run(async () => { await alertRules.remove(r.id); onChanged(); return true; })}>Delete</button>
    </li>
  );
}

/** `folded`: the form stays behind one "Add smart alert" button (stock pages, where the card sits in the side column). */
export function SmartAlerts({ ticker, onChanged, folded = false }: { ticker?: string; onChanged?: () => void; folded?: boolean }) {
  const rules = useApi("alert-rules", (signal) => alertRules.list(signal));
  const types = useApi("alert-rule-types", (signal) => alertRules.types(signal));
  const lists = useApi("watchlists", (signal) => watchlists.list(signal));
  const [type, setType] = useState("");
  const [threshold, setThreshold] = useState("");
  const [target, setTarget] = useState(ticker ?? "");
  const [scope, setScope] = useState<"ticker" | "watchlist">("ticker");
  const [listId, setListId] = useState<number | null>(null);
  const [email, setEmail] = useState(false);
  const [added, setAdded] = useState<string | null>(null);
  const [open, setOpen] = useState(!folded);
  const add = useAction();

  if (rules.error || types.error || !Array.isArray(types.data) || !Array.isArray(rules.data?.rules)
    || !Array.isArray(rules.data?.capabilities?.delivery_channels)) return null;
  if (!rules.data) return null;
  const all = types.data;
  // Market-wide types (scan alerts) are made from the Scanner, not this per-ticker form.
  const available = all.filter((t) => t.available && t.scope !== "market");
  const spec = available.find((t) => t.type === type) ?? available[0];
  const locked = all.filter((t) => !t.available);
  const emailAllowed = rules.data.capabilities.delivery_channels.includes("email");
  const listName = (id: number | null | undefined) => lists.data?.find((w) => w.id === id)?.name ?? "a watchlist";
  const mine = rules.data.rules.filter((r) => (ticker ? r.ticker === ticker.toUpperCase() && r.rule_type !== "SCAN_MATCH" : true));
  const full = rules.data.used >= rules.data.limit;
  const wl = listId ?? lists.data?.find((w) => w.is_default)?.id ?? lists.data?.[0]?.id ?? null;
  const changed = () => { rules.reload(); onChanged?.(); };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!spec) return;
    setAdded(null);
    const value = spec.threshold ? Number(threshold || spec.threshold.default) : undefined;
    const onList = !ticker && scope === "watchlist";
    const sym = (ticker ?? target).trim().toUpperCase();
    const ok = await add.run(async () => {
      await alertRules.create({
        rule_type: spec.type, enabled: true,
        ...(onList ? { watchlist_id: wl ?? undefined } : { ticker: sym }),
        ...(value !== undefined ? { threshold: value } : {}),
        delivery_channels: email && emailAllowed ? ["in_app", "email"] : ["in_app"],
      });
      return true;
    });
    if (ok) {
      if (folded) setOpen(false);
      setAdded(`${onList ? listName(wl) : sym}: ${describeRule({ rule_type: spec.type, threshold: value ?? null, value: null }, all)}`);
      setThreshold("");
      if (!ticker) setTarget("");
      changed();
    }
  };

  const canSubmit = !!spec && (ticker || scope === "watchlist" ? (ticker ? true : wl != null) : target.trim().length > 0);
  return (
    <Card title={ticker ? `Smart alerts on ${ticker}` : "Smart alerts"} id="smart"
      aside={`${rules.data.used} of ${rules.data.limit} alerts in use`}>
      {open && <p className="cap">Alerts on what HSF sees, not just price: score crossings, new setups, volume spikes, rank jumps
        and earnings heads-ups. Checked after each market scan.</p>}
      {mine.length > 0 ? (
        <ul className="alert-list">
          {mine.map((r) => (
            <RuleRow key={r.id} r={r} types={all} emailAllowed={emailAllowed} onChanged={changed}
              scope={r.rule_type === "SCAN_MATCH" ? "Scan alert" : r.ticker ?? `Every stock on ${listName(r.watchlist_id)}`} />
          ))}
        </ul>
      ) : <p className="cap">{ticker ? `No smart alerts on ${ticker} yet.` : "No smart alerts yet."}</p>}

      {full ? (
        <p className="cap">You&apos;re using every alert on your plan. Delete or turn one off to add another.</p>
      ) : spec && !open ? (
        <div className="stack-sm">
          {added && <p className="notice" role="status">Added: {added}.</p>}
          <div className="row-actions">
            <button type="button" className="btn" onClick={() => setOpen(true)} aria-expanded={false}>Add smart alert</button>
          </div>
        </div>
      ) : spec && (
        <form className="stack-sm" onSubmit={submit} aria-label="New smart alert">
          {!ticker && (
            <div className="row-actions" role="radiogroup" aria-label="Alert on">
              <label className="check"><input type="radio" name="smart-scope" checked={scope === "ticker"} onChange={() => setScope("ticker")} /><span>One ticker</span></label>
              <label className="check"><input type="radio" name="smart-scope" checked={scope === "watchlist"} disabled={!lists.data?.length}
                onChange={() => setScope("watchlist")} /><span>A whole watchlist</span></label>
            </div>
          )}
          <div className="inline-form">
            {!ticker && scope === "ticker" && (
              <label className="field"><span>Ticker</span>
                <input value={target} onChange={(e) => setTarget(e.target.value.toUpperCase())} maxLength={10} placeholder="AAPL"
                  autoCapitalize="characters" spellCheck={false} />
              </label>
            )}
            {!ticker && scope === "watchlist" && (
              <label className="field"><span>Watchlist</span>
                <select value={wl ?? ""} onChange={(e) => setListId(Number(e.target.value))}>
                  {(lists.data ?? []).map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
                </select>
              </label>
            )}
            <label className="field"><span>Alert me when</span>
              <select value={spec.type} onChange={(e) => { setType(e.target.value); setThreshold(""); }}>
                {available.map((t) => <option key={t.type} value={t.type}>{t.label.replace(/ \(days\)$/, "")}</option>)}
              </select>
            </label>
            {spec.threshold && (
              <label className="field"><span>{thresholdUnit(spec.type) ? `Value (${thresholdUnit(spec.type)})` : "Value"}</span>
                <input type="number" inputMode="decimal" min={spec.threshold.min} max={spec.threshold.max} step="any"
                  placeholder={String(spec.threshold.default ?? "")} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
              </label>
            )}
          </div>
          {emailAllowed && (
            <label className="check"><input type="checkbox" checked={email} onChange={(e) => setEmail(e.target.checked)} /><span>Also email me</span></label>
          )}
          <p className="cap">{spec.description}</p>
          <div className="row-actions">
            <button type="submit" className="btn" disabled={add.busy || !canSubmit}>{add.busy ? "Adding…" : "Add smart alert"}</button>
            {folded && <button type="button" className="btn" onClick={() => setOpen(false)}>Cancel</button>}
          </div>
          <ErrorLine error={add.error} />
          {added && <p className="notice" role="status">Added: {added}.</p>}
        </form>
      )}
      {open && locked.length > 0 && (
        <div className="stack-sm">
          <p className="cap">On a higher plan: {locked.map((t) => `${t.label.replace(/ \(days\)$/, "")} (${PLAN_OF[t.type] === "pro" ? "Pro" : "Premium"})`).join(", ")}.</p>
          <UpgradeButton plan={locked.some((t) => PLAN_OF[t.type] === "pro") ? "pro" : "premium"} />
        </div>
      )}
      {!ticker && <p className="cap">Scan alerts: set filters on the <Link href="/scanner">Scanner</Link>, then choose &ldquo;Alert me on new matches&rdquo;.</p>}
      {ticker && <p className="cap">Fired alerts show on <Link href="/alerts">Alerts</Link>.</p>}
    </Card>
  );
}
