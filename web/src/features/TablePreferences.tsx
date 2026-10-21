"use client";

import { useState } from "react";

import type { Schemas } from "@/api/client";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { saveWorkspace } from "@/hooks/useWorkspace";

type Col = NonNullable<Schemas["TablePreferences"]["columns"]>[number];
export const DEFAULT_COLUMNS = ["setup", "spark", "last", "change", "gap", "rvol", "prob"] as const;
/** Columns that can be moved. #, Ticker, Setup and HSF Score stay first. */
export const MOVABLE_COLUMNS = ["spark", "last", "change", "gap", "rvol", "prob"] as const;
const LABELS = { setup: "Setup", spark: "20 sessions", last: "Last price", change: "Change %", gap: "Gap %", rvol: "Relative volume", prob: "PreBreakout" };

/** Movable column ids in display order: the saved list's order (it is saved in order), then any not in it. */
export function columnOrder(saved: readonly string[]): Col[] {
  const inSaved = saved.filter((c): c is Col => (MOVABLE_COLUMNS as readonly string[]).includes(c));
  return [...inSaved, ...MOVABLE_COLUMNS.filter((c) => !inSaved.includes(c))];
}

export function TablePreferences({ initial, premium }: { initial: Schemas["TablePreferences"]; premium: boolean }) {
  const start = initial.columns ?? [...DEFAULT_COLUMNS];
  const [shown, setShown] = useState<Col[]>(start);
  const [order, setOrder] = useState<Col[]>(columnOrder(start));
  const [compact, setCompact] = useState(initial.compact ?? false);
  const [saved, setSaved] = useState(false);
  const act = useAction();
  const movable = order.filter((c) => c !== "prob" || premium);
  const move = (c: Col, by: -1 | 1) => {
    const at = movable.indexOf(c);
    const other = movable[at + by];
    if (!other) return;
    const next = [...order];
    next[order.indexOf(c)] = other;
    next[order.indexOf(other)] = c;
    setOrder(next); setSaved(false);
  };
  const check = (c: Col, on: boolean) => { setSaved(false); setShown(on ? [...shown, c] : shown.filter((x) => x !== c)); };
  const box = (c: Col) => <label><input type="checkbox" checked={shown.includes(c)} onChange={(e) => check(c, e.target.checked)} /> {LABELS[c]}</label>;
  return <details className="filter-panel"><summary>Table display</summary>
    <div className="stack-sm table-display-settings">
      <p className="cap">Ticker and HSF Score always stay visible. Use the arrows to change the column order. Display preferences apply to current scan tables and mobile cards.</p>
      <div className="row-actions">{box("setup")}</div>
      <ol className="column-order" aria-label="Column order">{movable.map((c, i) => <li key={c}>
        {box(c)}
        <span className="grow" />
        <button type="button" className="icon-btn" aria-label={`Move ${LABELS[c]} earlier`} disabled={i === 0} onClick={() => move(c, -1)}>↑</button>
        <button type="button" className="icon-btn" aria-label={`Move ${LABELS[c]} later`} disabled={i === movable.length - 1} onClick={() => move(c, 1)}>↓</button>
      </li>)}</ol>
      <label><input type="checkbox" checked={compact} onChange={(e) => { setCompact(e.target.checked); setSaved(false); }} /> Compact spacing</label>
      <div><button className="btn btn-sm" type="button" disabled={act.busy} onClick={() => void act.run(async () => {
        const columns = (["setup", ...order] as Col[]).filter((c) => shown.includes(c));
        await saveWorkspace({ table: { columns, compact } }); setSaved(true); return true;
      })}>Save display</button></div>
      {saved && <p role="status" className="cap">Display saved.</p>}
      <ErrorLine error={act.error} />
    </div>
  </details>;
}
