"use client";

import { useState } from "react";

import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { saveWorkspace, useWorkspace } from "@/hooks/useWorkspace";
import type { Schemas } from "@/api/client";

type View = Schemas["ScannerView"] | Schemas["CustomView"];

export function SavedResearchViews({ kind, current, onApply, disabled = false }: {
  kind: "scanner_views" | "custom_views"; current: View; onApply: (v: View) => void; disabled?: boolean;
}) {
  const prefs = useWorkspace();
  const act = useAction();
  const [name, setName] = useState("");
  const [selected, setSelected] = useState("");
  const views = prefs.data?.[kind] ?? [];
  const save = (next: View[]) => act.run(async () => {
    await saveWorkspace(kind === "scanner_views" ? { scanner_views: next as Schemas["ScannerView"][] }
      : { custom_views: next as Schemas["CustomView"][] });
    return true;
  });
  return <details className="saved-views filter-panel"><summary>Saved views{selected ? ` · ${selected}` : ""}</summary><div className="stack-sm table-display-settings">
    <div className="row-actions">
      <label className="inline-field">Saved views
        <select value={selected} disabled={disabled || !prefs.data || act.busy} onChange={(e) => {
          setSelected(e.target.value);
          const view = views.find((v) => v.name === e.target.value);
          if (view) onApply(view);
        }}><option value="">Choose a view</option>{views.map((v) => <option key={v.name}>{v.name}</option>)}</select>
      </label>
      <label className="field"><span className="sr-only">View name</span><input aria-label="View name" placeholder="Name this view" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} /></label>
      <button type="button" className="btn btn-sm" disabled={disabled || !prefs.data || act.busy || !name.trim() || (views.length >= 20 && !views.some((v) => v.name === name.trim()))}
        onClick={() => void save([...views.filter((v) => v.name !== name.trim()), { ...current, name: name.trim() }])}>Save view</button>
      {selected && <button type="button" className="btn btn-sm" disabled={act.busy || disabled}
        onClick={() => void save(views.filter((v) => v.name !== selected)).then((ok) => { if (ok) setSelected(""); })}>Delete view</button>}
    </div>
    <p className="cap">Up to 20 views saved to your account. Saving an existing name replaces it.</p>
    <ErrorLine error={prefs.error ?? act.error} />
  </div></details>;
}
