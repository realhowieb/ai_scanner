"use client";

// Premium: Claude's short coaching note on patterns in your own closed trades
// (POST /v1/ai/journal-coach), written from the journal's numbers only. Cached per
// journal state on the server, so asking again without new trades costs nothing.
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { AIText } from "@/components/AIText";
import { Card, ErrorLine, Locked } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useSession } from "@/session/SessionProvider";

export function JournalCoach() {
  const { can } = useSession();
  const act = useAction();
  const [out, setOut] = useState<Schemas["AIJournalCoach"] | null>(null);
  if (!can("can_ai_notes")) {
    return (
      <Card title="AI journal coach" id="coach" source="ai">
        <Locked title="The AI journal coach is part of Premium" plan="premium">
          A short note on patterns in your closed trades: how you do by HSF Score, holding time and source, and whether your wins outweigh your losses.
        </Locked>
      </Card>
    );
  }
  const run = () => void act.run(async () => {
    setOut(await unwrap(api.POST("/v1/ai/journal-coach")));
    return true;
  });
  return (
    <Card title="AI journal coach" id="coach" source="ai" aside="Premium">
      {out?.text ? <AIText text={out.text} />
        : out ? <p className="cap">Close at least {out.min_trades} trades for coaching (you have {out.closed}).</p>
        : <p className="cap">Claude reads your closed trades and points out patterns worth reviewing, from your journal&apos;s numbers only.</p>}
      <ErrorLine error={act.error} />
      <div className="row-actions">
        <button type="button" className="btn" onClick={run} disabled={act.busy}>{act.busy ? "Reading your journal…" : out?.text ? "Refresh coaching" : "Coach me"}</button>
      </div>
      <p className="cap">About your past results only. AI can be wrong; this is not investment advice.</p>
    </Card>
  );
}
