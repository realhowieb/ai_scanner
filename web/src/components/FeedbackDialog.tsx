"use client";

import { usePathname } from "next/navigation";
import { useId, useState } from "react";

import { api, unwrap } from "@/api/client";
import { Dialog } from "@/components/Dialog";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { captureUsage } from "@/lib/usageAnalytics";

type Kind = "problem" | "idea" | "praise" | "other";
const KINDS: [Kind, string][] = [["problem", "Something's wrong"], ["idea", "An idea"], ["praise", "Something I like"], ["other", "Other"]];

/** "Send feedback" from the account menu: goes to the HSF team with the page it was sent from. */
export function FeedbackDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const path = usePathname();
  const [kind, setKind] = useState<Kind>("problem");
  const [message, setMessage] = useState("");
  const [sent, setSent] = useState(false);
  const act = useAction();
  const id = useId();
  const close = () => { onClose(); setSent(false); setMessage(""); act.clear(); };
  const send = () => void act.run(async () => {
    await unwrap(api.POST("/v1/me/feedback", { body: {
      kind, message: message.trim(), path,
      build: (process.env.NEXT_PUBLIC_HSF_RELEASE || "").slice(0, 64).replace(/[^\w.-]/g, "") || undefined,
    } }));
    captureUsage("feedback_sent");
    setSent(true);
  });
  const ok = message.trim().length >= 3;

  return (
    <Dialog open={open} title="Send feedback" onClose={close}
      footer={sent ? <button type="button" className="btn btn-primary" onClick={close}>Done</button> : <>
        <button type="button" className="btn" onClick={close}>Cancel</button>
        <button type="button" className="btn btn-primary" onClick={send} disabled={!ok || act.busy}>{act.busy ? "Sending…" : "Send"}</button>
      </>}>
      {sent ? <p className="notice" role="status">Thanks, your message was sent. We read every one.</p> : (
        <form className="stack-sm" onSubmit={(e) => { e.preventDefault(); if (ok) send(); }}>
          <fieldset className="chips" aria-label="What is it about?">
            {KINDS.map(([k, label]) => (
              <label key={k} className={`chip${kind === k ? " chip-on" : ""}`}>
                <input type="radio" name={`${id}-kind`} value={k} checked={kind === k} onChange={() => setKind(k)} className="sr-only" />{label}
              </label>
            ))}
          </fieldset>
          <label className="field" htmlFor={`${id}-msg`}>
            <span className="cap">Your message</span>
            <textarea id={`${id}-msg`} className="input" rows={5} maxLength={2000} value={message} data-autofocus
              placeholder={kind === "problem" ? "What happened, and what did you expect?" : "Tell us what you think"}
              onChange={(e) => setMessage(e.target.value)} />
          </label>
          <p className="cap">We&apos;ll see your email address and the page you&apos;re on, so we can reply.</p>
          <ErrorLine error={act.error} />
        </form>
      )}
    </Dialog>
  );
}
