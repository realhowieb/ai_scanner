"use client";

// Premium research on one scan: Claude's summary of the top results in HSF Score order
// (POST /v1/ai/summary) and questions about the results (POST /v1/ai/chat). The
// conversation lives in this page only; the server uses the last 8 turns.
import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import { AIText } from "@/components/AIText";
import { Card, ErrorLine, Locked, ResearchNotice } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useSession } from "@/session/SessionProvider";

type Turn = { role: "user" | "assistant"; content: string };

const MAX_TURNS = 16;
const SUGGESTIONS = ["Which setups have the most confirming signals?", "Which names have earnings coming up?", "What do the top three have in common?"];

function Summary({ runId }: { runId?: number }) {
  const act = useAction();
  const [text, setText] = useState<string | null | undefined>(undefined);
  const run = () => void act.run(async () => {
    const out = await unwrap(api.POST("/v1/ai/summary", { body: runId ? { run_id: runId } : {} }));
    setText(out.text ?? null);
    return true;
  });
  return (
    <div className="stack-sm">
      {text ? <AIText text={text} />
        : text === null ? <p className="cap">Nothing to summarize: this scan has no ranked results.</p>
        : <p className="cap">Claude summarizes the leading research signals in HSF Score order, from this scan&apos;s data only.</p>}
      <ErrorLine error={act.error} />
      <div className="row-actions">
        <button type="button" className="btn" onClick={run} disabled={act.busy}>
          {act.busy ? "Writing…" : text ? "Write it again" : "Summarize this scan"}
        </button>
      </div>
    </div>
  );
}

function Chat({ runId }: { runId?: number }) {
  const act = useAction();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const ask = async (question: string) => {
    const text = question.trim().slice(0, 2000);
    if (!text || act.busy) return;
    const next: Turn[] = [...turns, { role: "user" as const, content: text }].slice(-MAX_TURNS);
    setTurns(next);
    setQ("");
    const ok = await act.run(async () => {
      const out = await unwrap(api.POST("/v1/ai/chat", { body: { messages: next, ...(runId ? { run_id: runId } : {}) } }));
      setTurns([...next, { role: "assistant", content: out.answer || "There are no results in this scan to answer from." }]);
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
    <div className="stack-sm">
      {turns.length === 0 ? (
        <div className="chips" role="group" aria-label="Example questions">
          {SUGGESTIONS.map((s) => <button key={s} type="button" className="chip" disabled={act.busy} onClick={() => void ask(s)}>{s}</button>)}
        </div>
      ) : (
        <ol className="chat" aria-live="polite" aria-label="Conversation">
          {turns.map((t, i) => (
            <li key={i} className={`chat-turn ${t.role}`}>
              <span className="sr-only">{t.role === "user" ? "You asked:" : "Claude:"}</span>
              {t.role === "assistant" ? <AIText text={t.content} /> : <p className="body-sm">{t.content}</p>}
            </li>
          ))}
          {act.busy && <li className="chat-turn assistant"><p className="cap" role="status">Thinking…</p></li>}
        </ol>
      )}
      <ErrorLine error={act.error} />
      <form className="row-actions" onSubmit={submit}>
        <label className="field grow"><span className="sr-only">Your question about this scan</span>
          <input value={q} maxLength={2000} placeholder="Ask about these results" onChange={(e) => setQ(e.target.value)} />
        </label>
        <button type="submit" className="btn btn-primary" disabled={act.busy || !q.trim()}>Ask</button>
        {turns.length > 0 && <button type="button" className="btn" disabled={act.busy} onClick={() => { setTurns([]); act.clear(); }}>New conversation</button>}
      </form>
    </div>
  );
}

/** The Premium AI card for the latest scan (no runId) or one of your saved scans. */
export function AIScanPanel({ runId }: { runId?: number }) {
  const { can } = useSession();
  if (!can("can_ai_notes")) {
    return (
      <Card title="AI research" id="ai">
        <Locked title="AI scan summaries and chat are part of Premium" plan="premium">Claude summarizes the evidence behind top results and answers research questions about this scan.</Locked>
      </Card>
    );
  }
  return (
    <Card title="AI research" id="ai" aside="Premium">
      <ResearchNotice compact />
      <Summary runId={runId} />
      <div className="divider" role="separator" />
      <p className="strong">Ask about this scan</p>
      <Chat runId={runId} />
      <p className="cap">AI commentary can be wrong. Treat it as a starting point for research, not an instruction to trade.</p>
    </Card>
  );
}
