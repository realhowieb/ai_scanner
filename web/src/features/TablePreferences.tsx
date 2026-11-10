"use client";

import { useState } from "react";

import type { Schemas } from "@/api/client";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { saveWorkspace } from "@/hooks/useWorkspace";

type Col = NonNullable<Schemas["TablePreferences"]["columns"]>[number];
export const DEFAULT_COLUMNS = ["setup", "sector", "spark", "last", "change", "gap", "rvol", "prob"] as const;
/** Columns that can be moved. #, Ticker, Setup and HSF Score stay first. */
export const MOVABLE_COLUMNS = ["sector", "spark", "last", "change", "gap", "rvol", "prob"] as const;
/** Columns added after people first saved their display: shown unless turned off (`hidden`),
 * and placed where they sit by default rather than after the saved ones. */
const NEW_COLUMNS: readonly Col[] = ["sector"];
const LABELS = { setup: "Setup", sector: "Sector", spark: "20 sessions", last: "Last price", change: "Change %", gap: "Gap %", rvol: "Relative volume", prob: "PreBreakout" };
type Table = { columns?: readonly string[] | null; hidden?: readonly string[] | null } | null | undefined;

/** The columns to show, in saved order: the saved list plus any new column not turned off. */
export function shownColumns(table: Table): Col[] {
  if (!table?.columns) return [...DEFAULT_COLUMNS];
  const saved = table.columns.filter((c): c is Col => (DEFAULT_COLUMNS as readonly string[]).includes(c));
  const added = NEW_COLUMNS.filter((c) => !saved.includes(c) && !(table.hidden ?? []).includes(c));
  return [...added, ...saved];
}

/** Movable column ids in display order: the saved list's order (it is saved in order), then any not in it.
 * A new column missing from the list goes where it sits by default. */
export function columnOrder(saved: readonly string[]): Col[] {
  const inSaved = saved.filter((c): c is Col => (MOVABLE_COLUMNS as readonly string[]).includes(c));
  const out = [...inSaved, ...MOVABLE_COLUMNS.filter((c) => !inSaved.includes(c) && !NEW_COLUMNS.includes(c))];
  for (const c of NEW_COLUMNS) {
    if (out.includes(c)) continue;
    const at = MOVABLE_COLUMNS.indexOf(c as (typeof MOVABLE_COLUMNS)[number]);
    const after = at > 0 ? out.indexOf(MOVABLE_COLUMNS[at - 1]!) : -1;
    out.splice(after + 1, 0, c);
  }
  return out;
}

export function TablePreferences({ initial, premium }: { initial: Schemas["TablePreferences"]; premium: boolean }) {
  const start = shownColumns(initial);
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
        const all = ["setup", ...order] as Col[];
        const columns = all.filter((c) => shown.includes(c));
        const hidden = all.filter((c) => !shown.includes(c));
        await saveWorkspace({ table: { columns, hidden, compact } }); setSaved(true); return true;
      })}>Save display</button></div>
      {saved && <p role="status" className="cap">Display saved.</p>}
      <ErrorLine error={act.error} />
    </div>
  </details>;
}
