"use client";

import { useState } from "react";
import Link from "@/components/Link";
import { alertRules, watchlists } from "@/api/userData";
import type { AlertRuleType, AlertType } from "@/api/userData";
import { TICKER_RE } from "@/components/AppShell";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { useSession } from "@/session/SessionProvider";
import { AlertForm, describeAlert, typesUnavailable } from "./AlertForm";

type Scope = "stock" | "watchlist" | "market";
const BASIC_RULES = new Set(["HSF_SCORE_ABOVE", "HSF_SCORE_BELOW", "HSF_SCORE_CROSS_ABOVE", "HSF_SCORE_CROSS_BELOW", "SETUP_APPEARED"]);

/** One entry point; each condition keeps its existing API and server validation. */
export function AlertComposer({ types, legacyFull = false, smartFull = false, onCreated }: { types: AlertType[]; legacyFull?: boolean; smartFull?: boolean; onCreated: (description: string) => void }) {
  const rules = useApi("alert-rule-types", (signal) => alertRules.types(signal));
  const [scope, setScope] = useState<Scope>("stock");
  const [choice, setChoice] = useState("");
  const legacy = types.filter((t) => scope === "stock" ? t.needs_ticker : scope === "watchlist" ? t.type === "watchlist" : t.type === "breakout");
  const smart = scope === "market" ? [] : (rules.data ?? []).filter((t) => t.scope !== "market");
  const options = [...legacy.map((t) => ({ id: `legacy:${t.type}`, label: t.label, available: true, basic: t.type === "price" || t.type === "watchlist" })),
    ...smart.map((t) => ({ id: `rule:${t.type}`, label: t.label, available: t.available, basic: BASIC_RULES.has(t.type) }))];
  const selected = options.find((o) => o.id === choice && o.available)
    ?? options.find((o) => o.available && o.basic) ?? options.find((o) => o.available);
  const smartType = smart.find((t) => `rule:${t.type}` === selected?.id);
  const legacyType = legacy.find((t) => `legacy:${t.type}` === selected?.id);
  return <div className="stack-sm">
    <label className="field"><span>Monitor</span>
      <select value={scope} onChange={(e) => { setScope(e.target.value as Scope); setChoice(""); }}>
        <option value="stock">One stock</option><option value="watchlist">A watchlist</option><option value="market">Market-wide setups</option>
      </select>
    </label>
    {selected && <label className="field"><span>Alert type</span>
      <select value={selected.id} onChange={(e) => setChoice(e.target.value)}>
        <optgroup label="Price, HSF Score and setups">{options.filter((o) => o.basic).map((o) => <option key={o.id} value={o.id} disabled={!o.available}>{o.label}{!o.available ? " · Not included in your plan" : ""}</option>)}</optgroup>
        <optgroup label="Advanced technical conditions">{options.filter((o) => !o.basic).map((o) => <option key={o.id} value={o.id} disabled={!o.available}>{o.label}{!o.available ? " · Not included in your plan" : ""}</option>)}</optgroup>
      </select>
    </label>}
    {scope === "market" && <p className="cap">For market-wide HSF Score, setup and earnings conditions, <Link href="/scanner">save a scan alert from the Scanner</Link>. Breakout Score monitoring is also available below.</p>}
    {scope !== "market" && !rules.data && (rules.error ? typesUnavailable(rules.error.status)
      ? <p className="cap">HSF Score and new-setup conditions are unavailable on this API. Other alert types remain available.</p>
      : <ErrorLine error={rules.error} /> : <p className="cap">Loading HSF conditions…</p>)}
    {(smartType ? smartFull : legacyFull) ? <p className="notice" role="status">This condition has reached your plan’s alert allowance. Manage existing alerts to free a slot.</p> : <>
    {smartType && <IntelligenceAlertForm key={`${scope}:${smartType.type}`} scope={scope === "watchlist" ? "watchlist" : "stock"} spec={smartType} onCreated={onCreated} />}
    {legacyType && <AlertForm key={`${scope}:${legacyType.type}`} types={[legacyType]} lockType prefill={{ type: legacyType.type }}
      onCreated={(a) => onCreated(describeAlert(a))} />}
    </>}
    {!selected && <p className="cap">No conditions are available for this scope.</p>}
  </div>;
}

function IntelligenceAlertForm({ scope, spec, onCreated }: { scope: "stock" | "watchlist"; spec: AlertRuleType; onCreated: (description: string) => void }) {
  const { can } = useSession();
  const lists = useApi(scope === "watchlist" ? "watchlists" : null, (signal) => watchlists.list(signal));
  const [ticker, setTicker] = useState("");
  const [list, setList] = useState("");
  const [threshold, setThreshold] = useState(spec.threshold?.default != null ? String(spec.threshold.default) : "");
  const [value, setValue] = useState("");
  const [email, setEmail] = useState(false);
  const [invalid, setInvalid] = useState<string | null>(null);
  const act = useAction();
  const target = scope === "stock" ? ticker.trim().toUpperCase() : lists.data?.find((w) => String(w.id) === list)?.name;
  const preview = `${target || (scope === "stock" ? "Choose a ticker" : "Choose a watchlist")}: ${spec.label}${spec.threshold ? ` ${threshold || "…"}` : ""}${value ? ` (${value})` : ""}`;
  return <form className="stack-sm alert-form" noValidate onSubmit={async (e) => {
    e.preventDefault();
    const n = Number(threshold);
    const problem = scope === "stock" && !TICKER_RE.test(ticker.trim()) ? "Enter a ticker symbol like AAPL."
      : scope === "watchlist" && !lists.data?.some((w) => String(w.id) === list) ? "Choose a watchlist."
      : spec.threshold && (!threshold.trim() || !Number.isFinite(n) || n < spec.threshold.min || n > spec.threshold.max)
        ? `Enter a value from ${spec.threshold.min} to ${spec.threshold.max}.` : null;
    setInvalid(problem);
    if (problem) return;
    const ok = await act.run(() => alertRules.create({ rule_type: spec.type, enabled: true,
      ...(scope === "stock" ? { ticker: ticker.trim().toUpperCase() } : { watchlist_id: Number(list) }),
      ...(spec.threshold ? { threshold: n } : {}), ...(spec.takes_value && value.trim() ? { value: value.trim() } : {}),
      delivery_channels: email ? ["in_app", "email"] : ["in_app"],
    }));
    if (ok) onCreated(preview);
  }}>
    <p className="cap">{spec.description}</p>
    {scope === "stock" ? <label className="field"><span>Ticker</span><input value={ticker} maxLength={10} autoComplete="off" placeholder="AAPL" onChange={(e) => setTicker(e.target.value)} /></label>
      : <label className="field"><span>Watchlist</span><select value={list} onChange={(e) => setList(e.target.value)} disabled={!lists.data}>
        <option value="">Choose a watchlist</option>{lists.data?.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
      </select></label>}
    {scope === "watchlist" && lists.data?.length === 0 && <p className="cap">Create a watchlist before using this condition.</p>}
    <ErrorLine error={lists.error} />
    {spec.threshold && <label className="field"><span>Value</span><input inputMode="decimal" type="number" step="any" min={spec.threshold.min} max={spec.threshold.max} value={threshold} onChange={(e) => setThreshold(e.target.value)} /></label>}
    {spec.takes_value && <label className="field"><span>Setup name (optional)</span><input value={value} maxLength={40} placeholder="Any ranked setup" onChange={(e) => setValue(e.target.value)} /></label>}
    {can("can_email_alerts") && <label className="check"><input type="checkbox" checked={email} onChange={(e) => setEmail(e.target.checked)} /><span>Also email me</span></label>}
    <p className="notice" aria-label="Rule preview">{preview}</p>
    <p className="cap">Checked after market scans. {spec.kind === "transition" ? "The first scan sets a baseline; a later transition can trigger this rule." : "Repeated matches follow the rule’s cooldown."} Email receipt depends on delivery status.</p>
    {invalid && <p className="form-error" role="alert">{invalid}</p>}
    <ErrorLine error={act.error} />
    <button type="submit" className="btn btn-primary" disabled={act.busy || (scope === "watchlist" && !lists.data?.length)}>{act.busy ? "Creating…" : "Create alert"}</button>
  </form>;
}
