"use client";

// Premium: Ask HSF (POST /v1/ai/ask). One conversation about your watchlists, the latest
// market scan, earnings, news headlines and HSF's track record. The server adds fresh
// facts for each question; the conversation lives in this page only.
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import { AIText } from "@/components/AIText";
import { Card, ErrorLine, Locked } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { captureUsage } from "@/lib/usageAnalytics";
import { useSession } from "@/session/SessionProvider";

type Turn = { role: "user" | "assistant"; content: string };

const MAX_TURNS = 16;
export const ASK_SUGGESTIONS = [
  "What changed on my watchlist since the last scan?",
  "Which of my stocks report earnings this week?",
  "What do today's top setups have in common?",
  "Is there news on my strongest names?",
];

export function AskHSFView() {
  const { can } = useSession();
  const act = useAction();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");

  if (!can("can_ai_notes")) {
    return (
      <div className="stack">
        <section className="page-head"><div><h1 className="h1">Ask HSF</h1></div></section>
        <Card>
          <Locked title="Ask HSF is part of Premium" plan="premium">
            Ask anything about your watchlists, today&apos;s scan, earnings and news, and get an answer written from HSF&apos;s own data.
          </Locked>
        </Card>
      </div>
    );
  }

  const ask = async (question: string) => {
    const text = question.trim().slice(0, 2000);
    if (!text || act.busy) return;
    const next: Turn[] = [...turns, { role: "user" as const, content: text }].slice(-MAX_TURNS);
    setTurns(next);
    setQ("");
    const ok = await act.run(async () => {
      const out = await unwrap(api.POST("/v1/ai/ask", { body: { messages: next } }));
      captureUsage("ask_hsf_asked");
      setTurns([...next, { role: "assistant", content: out.answer || "I couldn't find anything in HSF's data to answer that." }]);
      return true;
    });
    if (!ok) {
      setTurns(turns);
      setQ(text);
    }
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    void ask(q);
  };

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Ask HSF</h1>
          <p className="cap">Questions about your watchlists, the latest scan, earnings, news and HSF&apos;s track record. Premium.</p>
        </div>
      </section>
      <Card title="Conversation" aside="Premium">
        {turns.length === 0 ? (
          <div className="chips" role="group" aria-label="Example questions">
            {ASK_SUGGESTIONS.map((s) => <button key={s} type="button" className="chip" disabled={act.busy} onClick={() => void ask(s)}>{s}</button>)}
          </div>
        ) : (
          <ol className="chat" aria-live="polite" aria-label="Conversation">
            {turns.map((t, i) => (
              <li key={i} className={`chat-turn ${t.role}`}>
                <span className="sr-only">{t.role === "user" ? "You asked:" : "HSF:"}</span>
                {t.role === "assistant" ? <AIText text={t.content} /> : <p className="body-sm">{t.content}</p>}
              </li>
            ))}
            {act.busy && <li className="chat-turn assistant"><p className="cap" role="status">Looking through HSF&apos;s data…</p></li>}
          </ol>
        )}
        <ErrorLine error={act.error} />
        <form className="row-actions" onSubmit={submit}>
          <label className="field grow"><span className="sr-only">Your question</span>
            <input value={q} maxLength={2000} placeholder="Ask about your stocks, e.g. how does $NVDA look today?" onChange={(e) => setQ(e.target.value)} />
          </label>
          <button type="submit" className="btn btn-primary" disabled={act.busy || !q.trim()}>Ask</button>
          {turns.length > 0 && <button type="button" className="btn" disabled={act.busy} onClick={() => { setTurns([]); act.clear(); }}>New conversation</button>}
        </form>
        <p className="cap">Answers are written by Claude from HSF&apos;s data only and can be wrong. Research only, not investment advice. Name a stock with $ (like $AAPL) to pull its news and track record.</p>
      </Card>
    </div>
  );
}
