"use client";

// Premium: Claude's short rundown of what changed on one watchlist since the previous
// market scan, from the saved scan's facts only (no news, no quotes). The last rundown
// (from the morning or Sunday email, or a refresh) shows on load without an AI call.
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { AIText } from "@/components/AIText";
import { Card, ErrorLine, Locked } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

export function WatchlistAINote({ id, name, empty }: { id: number; name: string; empty: boolean }) {
  const { can } = useSession();
  const act = useAction();
  const [note, setNote] = useState<{ id: number; v: Schemas["AIText"] } | null>(null);
  const allowed = can("can_ai_notes");
  const saved = useApi(allowed && !empty ? `wl-ai-saved:${id}` : null, (signal) =>
    unwrap(api.GET("/v1/ai/watchlists/{watchlist_id}/note", { params: { path: { watchlist_id: id } }, signal })));
  const fresh = note && note.id === id ? note.v : null;
  const mine = fresh ?? (saved.data?.text ? saved.data : null);
  if (!allowed) {
    return (
      <Card title="AI rundown" id="wl-ai" source="ai">
        <Locked title="AI watchlist rundowns are part of Premium" plan="premium">
          A short plain-English summary of what changed on your list since the last scan: score and rank moves, new setups, volume and earnings this week.
        </Locked>
      </Card>
    );
  }
  if (empty) return null;
  const run = () => void act.run(async () => {
    const v = await unwrap(api.POST("/v1/ai/watchlists/{watchlist_id}/note", { params: { path: { watchlist_id: id } } }));
    setNote({ id, v });
    return true;
  });
  return (
    <Card title="AI rundown" id="wl-ai" source="ai" aside={mine?.snapshot_time ? `Scan ${etTime(mine.snapshot_time)}` : "Premium"}>
      {!fresh && mine?.auto && <p className="cap" data-testid="auto-written">Written automatically{mine.written_at ? ` ${etTime(mine.written_at)}` : ""}. A new one arrives with your morning email and every Sunday.</p>}
      {mine ? (mine.text ? <AIText text={mine.text} /> : <p className="cap">Nothing to sum up yet: no market scan covers {name}.</p>)
        : <p className="cap">What changed on {name} since the previous scan, written by Claude from HSF&apos;s scan data only.</p>}
      <ErrorLine error={act.error} />
      <div className="row-actions">
        <button type="button" className="btn" onClick={run} disabled={act.busy}>{act.busy ? "Writing…" : mine ? "Refresh rundown" : "Write AI rundown"}</button>
      </div>
      <p className="cap">AI commentary can be wrong. Research only, not investment advice.</p>
    </Card>
  );
}
