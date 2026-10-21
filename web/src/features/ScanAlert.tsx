"use client";

// Scan alerts (Pro): save the Scanner's current filters as a market-wide alert rule
// (SCAN_MATCH). The server checks it after every market scan and fires when stocks that
// weren't matching on the previous scan now match. Optional price and volume limits.
import Link from "next/link";
import { useState } from "react";
import type { FormEvent } from "react";

import { alertRules } from "@/api/userData";
import { Dialog } from "@/components/Dialog";
import { ErrorLine, Locked, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";

export type ScanEarnings = "today" | "tomorrow" | "week";
export type ScanFilters = { name?: string; setup?: string; min_price?: number; max_price?: number; min_rvol?: number; earnings?: ScanEarnings };

const EARNINGS_NAMES: Record<ScanEarnings, string> = {
  today: "reporting today", tomorrow: "reporting tomorrow", week: "reporting in the next 7 days",
};

const SETUP_NAMES: Record<string, string> = {
  breakout: "Breakout", gapper: "Gapper", golden_cross: "Golden cross", gainer: "Gainer", prebreakout: "PreBreakout",
};

export function parseScanFilters(value: string | null | undefined): ScanFilters {
  if (!value) return {};
  try {
    const v = JSON.parse(value) as unknown;
    return v && typeof v === "object" && !Array.isArray(v) ? (v as ScanFilters) : {};
  } catch {
    return {};
  }
}

/** "Breakouts: HSF 75+, Breakout, $5 to $50, 2x volume" */
export function describeScanAlert(threshold: number | null | undefined, value: string | null | undefined): string {
  const f = parseScanFilters(value);
  const parts = [`HSF ${threshold ?? 0}+`];
  if (f.setup) parts.push(SETUP_NAMES[f.setup] ?? f.setup);
  if (f.earnings) parts.push(EARNINGS_NAMES[f.earnings] ?? f.earnings);
  if (f.min_price != null && f.max_price != null) parts.push(`$${f.min_price} to $${f.max_price}`);
  else if (f.min_price != null) parts.push(`$${f.min_price}+`);
  else if (f.max_price != null) parts.push(`under $${f.max_price}`);
  if (f.min_rvol != null) parts.push(`${f.min_rvol}x volume`);
  return `${f.name ? `${f.name}: ` : ""}${parts.join(", ")}`;
}

const num = (s: string): number | undefined => {
  const v = Number(s);
  return s.trim() !== "" && Number.isFinite(v) && v >= 0 ? v : undefined;
};

export function ScanAlertButton({ setup, minScore, setupLabel, earnings }: { setup: string; minScore: number; setupLabel?: string; earnings?: ScanEarnings }) {
  const [open, setOpen] = useState(false);
  const types = useApi(open ? "alert-rule-types" : null, (signal) => alertRules.types(signal));
  const rules = useApi(open ? "alert-rules" : null, (signal) => alertRules.list(signal));
  const [name, setName] = useState("");
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minRvol, setMinRvol] = useState("");
  const [email, setEmail] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const act = useAction();
  const spec = Array.isArray(types.data) ? types.data.find((t) => t.type === "SCAN_MATCH") : undefined;
  const full = !!rules.data && rules.data.used >= rules.data.limit;
  const emailAllowed = !!rules.data?.capabilities.delivery_channels.includes("email");
  const close = () => { setOpen(false); setDone(null); };
  const filters: ScanFilters = {
    ...(name.trim() ? { name: name.trim() } : {}), ...(setup ? { setup } : {}), ...(earnings ? { earnings } : {}),
    ...(num(minPrice) !== undefined ? { min_price: num(minPrice) } : {}),
    ...(num(maxPrice) !== undefined ? { max_price: num(maxPrice) } : {}),
    ...(num(minRvol) !== undefined ? { min_rvol: num(minRvol) } : {}),
  };
  const value = Object.keys(filters).length ? JSON.stringify(filters) : undefined;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const ok = await act.run(async () => {
      await alertRules.create({ rule_type: "SCAN_MATCH", threshold: minScore, ...(value ? { value } : {}), enabled: true,
        delivery_channels: email && emailAllowed ? ["in_app", "email"] : ["in_app"] });
      return true;
    });
    if (ok) { setDone(describeScanAlert(minScore, value)); rules.reload(); }
  };

  return (
    <>
      <button type="button" className="btn btn-sm" onClick={() => setOpen(true)}>Alert me on new matches</button>
      <Dialog open={open} title="Scan alert" onClose={close}>
        {done ? (
          <div className="stack-sm">
            <p className="notice" role="status">Created: {done}. HSF checks it after every market scan.</p>
            <div className="row-actions"><button type="button" className="btn" onClick={close} data-autofocus>Done</button><Link href="/alerts">See all alerts →</Link></div>
          </div>
        ) : types.error || rules.error ? <ErrorLine error={types.error ?? rules.error} />
          : !types.data || !rules.data ? <p className="cap">Loading…</p>
          : !spec ? <p className="notice" role="note">Scan alerts aren&apos;t available yet.</p>
          : !spec.available ? (
            <div className="stack-sm">
              <Locked title="Scan alerts are part of Pro" plan="pro">Save these Scanner filters and get told when new stocks match after each scan.</Locked>
              <UpgradeButton plan="pro" />
            </div>
          ) : full ? (
            <div className="stack-sm">
              <p className="strong">You&apos;re using all {rules.data.limit} alerts on your plan.</p>
              <p className="cap">Delete or turn one off on the <Link href="/alerts">Alerts</Link> page to add this one.</p>
            </div>
          ) : (
            <form className="stack-sm" onSubmit={submit} aria-label="New scan alert">
              <p className="cap">Watches the whole market for {setupLabel && setup ? `${setupLabel} setups` : "ranked setups"}{earnings ? ` ${EARNINGS_NAMES[earnings]}` : ""} at HSF Score {minScore}+. After each scan you get one alert listing the stocks that newly match. Using {rules.data.used} of {rules.data.limit} alerts.</p>
              <label className="field"><span>Name (optional)</span>
                <input value={name} maxLength={60} placeholder="My breakouts" onChange={(e) => setName(e.target.value)} data-autofocus /></label>
              <div className="inline-form">
                <label className="field"><span>Min price ($)</span><input type="number" inputMode="decimal" min={0} step="any" value={minPrice} onChange={(e) => setMinPrice(e.target.value)} /></label>
                <label className="field"><span>Max price ($)</span><input type="number" inputMode="decimal" min={0} step="any" value={maxPrice} onChange={(e) => setMaxPrice(e.target.value)} /></label>
                <label className="field"><span>Min relative volume (x)</span><input type="number" inputMode="decimal" min={0} step="any" value={minRvol} onChange={(e) => setMinRvol(e.target.value)} /></label>
              </div>
              {emailAllowed && <label className="check"><input type="checkbox" checked={email} onChange={(e) => setEmail(e.target.checked)} /><span>Also email me</span></label>}
              <p className="cap">{describeScanAlert(minScore, value)}</p>
              <div className="row-actions"><button type="submit" className="btn" disabled={act.busy}>{act.busy ? "Saving…" : "Save scan alert"}</button></div>
              <ErrorLine error={act.error} />
            </form>
          )}
      </Dialog>
    </>
  );
}
