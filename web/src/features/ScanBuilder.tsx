"use client";

// Premium: describe a scan in plain English and Claude fills in the Custom scan form
// (POST /v1/ai/scan-builder). Nothing runs until you press Start; plan rules still apply.
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useSession } from "@/session/SessionProvider";

export function ScanBuilder({ onApply, disabled = false }: { onApply: (v: Schemas["AIScanView"]) => void; disabled?: boolean }) {
  const { can } = useSession();
  const act = useAction();
  const [text, setText] = useState("");
  const [explain, setExplain] = useState<string | null>(null);
  if (!can("can_ai_notes")) return null;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const t = text.trim();
    if (t.length < 3 || act.busy) return;
    void act.run(async () => {
      const out = await unwrap(api.POST("/v1/ai/scan-builder", { body: { text: t } }));
      onApply(out.view);
      setExplain(out.explanation ?? "Settings filled in below. Check them, then start the scan.");
      return true;
    });
  };
  return (
    <form className="card stack-sm" onSubmit={submit} aria-label="Describe a scan">
      <div className="row-actions">
        <label className="field grow"><span>Describe your scan <span className="pill pill-gold">Premium AI</span></span>
          <input value={text} maxLength={500} disabled={disabled}
            placeholder="e.g. NASDAQ gappers over $5 with unusual volume, top 50"
            onChange={(e) => setText(e.target.value)} />
        </label>
        <button type="submit" className="btn" disabled={disabled || act.busy || text.trim().length < 3}>{act.busy ? "Filling in…" : "Fill in settings"}</button>
      </div>
      <ErrorLine error={act.error} />
      {explain && <p className="cap" role="status">{explain} Nothing runs until you press Start.</p>}
    </form>
  );
}
