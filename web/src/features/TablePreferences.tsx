"use client";

import { useState } from "react";

import type { Schemas } from "@/api/client";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { saveWorkspace } from "@/hooks/useWorkspace";

export const DEFAULT_COLUMNS = ["setup", "spark", "last", "change", "gap", "rvol", "prob"] as const;
const LABELS = { setup: "Setup", spark: "20 sessions", last: "Last price", change: "Change %", gap: "Gap %", rvol: "Relative volume", prob: "PreBreakout" };
export function TablePreferences({ initial, premium }: { initial: Schemas["TablePreferences"]; premium: boolean }) {
  const [columns, setColumns] = useState(initial.columns ?? [...DEFAULT_COLUMNS]);
  const [compact, setCompact] = useState(initial.compact ?? false);
  const [saved, setSaved] = useState(false);
  const act = useAction();
  return <details className="filter-panel"><summary>Table display</summary>
    <div className="stack-sm table-display-settings">
      <p className="cap">Ticker and HSF Score always stay visible. Display preferences apply to current scan tables and mobile cards.</p>
      <div className="row-actions">{DEFAULT_COLUMNS.filter((c) => c !== "prob" || premium).map((c) => <label key={c}>
        <input type="checkbox" checked={columns.includes(c)} onChange={(e) => { setSaved(false); setColumns(e.target.checked ? [...columns, c] : columns.filter((x) => x !== c)); }} /> {LABELS[c]}
      </label>)}</div>
      <label><input type="checkbox" checked={compact} onChange={(e) => { setCompact(e.target.checked); setSaved(false); }} /> Compact spacing</label>
      <div><button className="btn btn-sm" type="button" disabled={act.busy} onClick={() => void act.run(async () => {
        await saveWorkspace({ table: { columns, compact } }); setSaved(true); return true;
      })}>Save display</button></div>
      {saved && <p role="status" className="cap">Display saved.</p>}
      <ErrorLine error={act.error} />
    </div>
  </details>;
}
