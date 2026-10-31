"use client";

// Admin: Founding 100 refund requests, one decision at a time (no bulk actions). Approving
// records the decision; the refund goes to Stripe only when FOUNDING_REFUNDS_ENABLED is on,
// and access changes only when Stripe confirms it.
import { useId, useState } from "react";

import { cents, founderBenefits, founderBriefing, REFUND_STATUS_LABEL, refunds } from "@/api/founding";
import type { AdminRefundRequest, BriefingPreview, RefundBasis, RefundStatus } from "@/api/founding";
import { ConfirmDialog, Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etDate, etTime } from "@/lib/format";

const BASES: [RefundBasis, string][] = [
  ["guarantee", "14-day guarantee"], ["billing_error", "Billing error"], ["service_failure", "Service failure"],
  ["legal_requirement", "Required by law"], ["discontinuation", "Permanent discontinuation (prorated)"],
  ["exception", "Admin exception"],
];
const FILTERS: [RefundStatus | "", string][] = [["requested", "Waiting"], ["", "All"], ["approved", "Approved"],
  ["initiated", "Started"], ["completed", "Refunded"], ["failed", "Failed"], ["denied", "Denied"]];
const TONE = { requested: "warn", approved: "up", initiated: "up", completed: "up", denied: "down", failed: "down" } as const;

function Decide({ id, onClose, onChanged }: { id: number; onClose: () => void; onChanged: () => void }) {
  const state = useApi(`admin-founding-refund-${id}`, (signal) => refunds.detail(id, signal));
  const d = state.data;
  const r = d?.request;
  const [basis, setBasis] = useState<RefundBasis | "">("");
  const [amount, setAmount] = useState("");
  const [reason, setReason] = useState("");
  const [stopped, setStopped] = useState("");
  const [quote, setQuote] = useState<number | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const act = useAction();
  const chosen: RefundBasis | "" = basis || (r?.status === "approved" && r.basis ? r.basis : r?.within_guarantee ? "guarantee" : "");
  const stoppedIso = stopped ? `${stopped}T00:00:00Z` : "";

  const done = (msg: string) => { setNote(msg); onChanged(); state.reload(); };
  const approve = () => void act.run(async () => {
    if (!chosen) return false;
    const dollars = amount.trim() ? Math.round(Number(amount) * 100) : undefined;
    const out = await refunds.approve(id, { basis: chosen, reason, ...(dollars ? { amount_cents: dollars } : {}),
      ...(chosen === "discontinuation" && stoppedIso ? { stopped_at: stoppedIso } : {}) });
    done(out.note ?? "Approved.");
    return true;
  });
  const deny = () => void act.run(async () => {
    await refunds.deny(id, reason);
    done("Denied. The member has been emailed the reason.");
    return true;
  });
  const prorate = () => void act.run(async () => {
    setQuote((await refunds.prorated(id, stoppedIso)).amount_cents);
    return true;
  });

  return (
    <Dialog open title={`Refund request #${id}`} onClose={onClose}>
      {state.error && !d ? <ErrorState error={state.error} onRetry={state.reload} what="this request" />
        : !d || !r ? <Skeleton rows={3} label="Loading the request" /> : (
          <div className="stack-sm">
            <dl className="kv">
              <div><dt>Member</dt><dd>{r.user_id}{r.number ? ` · #${r.number}` : ""} · {r.membership_status}</dd></div>
              <div><dt>Payment</dt><dd>{r.kind === "upgrade" ? "Upgrade" : "Purchase"} {cents(r.paid_cents)} · paid {etDate(r.paid_at)}{r.refunded_cents ? ` · ${cents(r.refunded_cents)} refunded` : ""}</dd></div>
              <div><dt>Asked</dt><dd>{etTime(r.requested_at)} · {r.within_guarantee ? "within 14 days" : "after 14 days"}</dd></div>
              <div><dt>Start by</dt><dd>{etDate(r.due_at)}</dd></div>
              <div><dt>Status</dt><dd><Pill tone={TONE[r.status]}>{REFUND_STATUS_LABEL[r.status]}</Pill></dd></div>
              {r.member_reason && <div><dt>Member says</dt><dd>{r.member_reason}</dd></div>}
              {r.failure_reason && <div><dt>Last error</dt><dd>{r.failure_reason}</dd></div>}
              <div><dt>Policies accepted</dt><dd>{d.accepted_policies.length ? d.accepted_policies.map((a) => `${a.policy_type} v${a.version}`).join(", ") : "none on file"}</dd></div>
              <div><dt>Payment ref</dt><dd className="cap">{r.payment_reference}</dd></div>
            </dl>
            {!d.refunds_live && <p className="banner" role="note">Refunds are switched off (FOUNDING_REFUNDS_ENABLED). Approving records the decision; nothing is sent to Stripe.</p>}
            {note && <p className="notice" role="status">{note}</p>}
            {(r.status === "requested" || r.status === "approved") && (
              <div className="stack-sm">
                {r.status === "requested" && (
                  <label className="field"><span>Why approve</span>
                    <select value={chosen} onChange={(e) => { setBasis(e.target.value as RefundBasis); setQuote(null); }}>
                      <option value="">Choose…</option>
                      {BASES.filter(([b]) => b !== "guarantee" || r.within_guarantee).map(([b, label]) => (
                        <option key={b} value={b} disabled={b === "discontinuation" && !d.prorated_live}>{label}{b === "discontinuation" && !d.prorated_live ? " (off)" : ""}</option>
                      ))}
                    </select>
                  </label>
                )}
                {r.status === "requested" && chosen === "discontinuation" && (
                  <div className="row-actions">
                    <label className="field"><span>Service stopped on</span><input type="date" value={stopped} onChange={(e) => { setStopped(e.target.value); setQuote(null); }} /></label>
                    <button type="button" className="btn btn-sm" disabled={!stopped || act.busy} onClick={prorate}>Work out refund</button>
                    {quote !== null && <strong>{cents(quote)}</strong>}
                  </div>
                )}
                {r.status === "requested" && chosen !== "discontinuation" && (
                  <label className="field"><span>Amount in dollars (blank = {cents(r.paid_cents - r.refunded_cents)})</span>
                    <input inputMode="decimal" value={amount} onChange={(e) => setAmount(e.target.value)} />
                  </label>
                )}
                <label className="field"><span>{r.status === "requested" ? "Note (required to deny; sent to the member)" : "Note"}</span>
                  <textarea rows={2} maxLength={1000} value={reason} onChange={(e) => setReason(e.target.value)} />
                </label>
                <ErrorLine error={act.error} />
                <div className="row-actions end">
                  {r.status === "requested" && <button type="button" className="btn btn-danger" disabled={act.busy || !reason.trim()} onClick={deny}>Deny</button>}
                  <button type="button" className="btn btn-primary" disabled={act.busy || !chosen || (chosen === "discontinuation" && !stopped)} onClick={approve}>
                    {r.status === "approved" ? "Retry sending to Stripe" : d.refunds_live ? "Approve and send refund" : "Approve"}
                  </button>
                </div>
              </div>
            )}
            <h3 className="h3">Audit trail</h3>
            <ul className="stack-xs">{d.audit.map((a, i) => <li key={`${a.at}-${i}`} className="cap">{etTime(a.at)} · {a.actor} · {a.action.replaceAll("_", " ")}</li>)}</ul>
          </div>
        )}
    </Dialog>
  );
}

export function FoundingRefundsTab() {
  const [status, setStatus] = useState<RefundStatus | "">("requested");
  const [open, setOpen] = useState<number | null>(null);
  const state = useApi(`admin-founding-refunds-${status}`, (signal) => refunds.list(status || undefined, signal));
  const rows: AdminRefundRequest[] = state.data?.requests ?? [];
  return (
    <Card title="Founding 100 refunds">
      <div className="stack-sm">
        <div className="seg-group" role="group" aria-label="Show">
          {FILTERS.map(([k, label]) => (
            <button key={k || "all"} type="button" aria-pressed={status === k} className={`seg${status === k ? " on" : ""}`} onClick={() => setStatus(k)}>{label}</button>
          ))}
        </div>
        {state.data && !state.data.refunds_live && <p className="cap">Refunds switched off: approvals are recorded but not sent to Stripe.</p>}
        {state.error && !state.data ? <ErrorState error={state.error} onRetry={state.reload} what="refund requests" />
          : !state.data ? <Skeleton rows={3} label="Loading refund requests" />
            : rows.length === 0 ? <Empty title="Nothing here.">No refund requests with this status.</Empty> : (
              <div className="table-wrap"><table className="table">
                <thead><tr><th scope="col">Asked</th><th scope="col">Member</th><th scope="col">Payment</th><th scope="col">Status</th><th scope="col"><span className="sr-only">Open</span></th></tr></thead>
                <tbody>{rows.map((r) => (
                  <tr key={r.id}>
                    <td className="cap">{etDate(r.requested_at)}<div>{r.within_guarantee ? "within 14 days" : "after 14 days"}</div></td>
                    <td>{r.user_id}</td>
                    <td>{r.kind === "upgrade" ? "Upgrade" : r.tier === "founder_elite" ? "Elite" : "Pro"} {cents(r.paid_cents)}</td>
                    <td><Pill tone={TONE[r.status]}>{REFUND_STATUS_LABEL[r.status]}</Pill></td>
                    <td><button type="button" className="btn btn-sm" onClick={() => setOpen(r.id)}>Review</button></td>
                  </tr>
                ))}</tbody>
              </table></div>
            )}
      </div>
      {open !== null && <Decide id={open} onClose={() => setOpen(null)} onChanged={state.reload} />}
    </Card>
  );
}

export function FounderPollCard() {
  const state = useApi("admin-founder-poll", (signal) => founderBenefits.pollResults(signal));
  const d = state.data;
  return (
    <Card title="Founder poll">
      {state.error && !d ? <ErrorState error={state.error} onRetry={state.reload} what="the founder poll" />
        : !d ? <Skeleton rows={2} label="Loading the founder poll" />
          : d.options.length === 0 ? <p className="cap">No poll is set (HSF_FOUNDER_POLL is empty).</p> : (
            <div className="stack-xs">
              <p className="cap">{d.total} founder vote{d.total === 1 ? "" : "s"}. Change the options with HSF_FOUNDER_POLL.</p>
              <ul className="stack-xs">{[...d.options].sort((a, b) => (b.votes ?? 0) - (a.votes ?? 0)).map((o) => (
                <li key={o.key}><strong>{o.votes ?? 0}</strong> <span>{o.label}</span></li>
              ))}</ul>
            </div>
          )}
    </Card>
  );
}

/** The monthly Founder Briefing: write what shipped, preview it, then send it once. */
export function FounderBriefingCard() {
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<BriefingPreview | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const look = useAction();
  const send = useAction();
  const id = useId();
  const doPreview = () => void look.run(async () => { setPreview(await founderBriefing.preview(note)); return true; });
  const doSend = () => void send.run(async () => {
    const out = await founderBriefing.send(note);
    setResult(`Sent to ${out.sent} founder${out.sent === 1 ? "" : "s"}${out.failed ? `, ${out.failed} failed` : ""}.`);
    setConfirming(false);
    setPreview(await founderBriefing.preview(note));
    return true;
  });
  const ready = note.trim().length >= 20;
  return (
    <Card title="Founder Briefing" aside={preview ? `${preview.recipients} active founder${preview.recipients === 1 ? "" : "s"}` : undefined}>
      <div className="stack-sm">
        <p className="cap">Once a month. Each founder gets your note, the poll results, early access and their own founder value. Never sent automatically.</p>
        <label className="field" htmlFor={id}>
          <span className="cap">What shipped this month (blank line between paragraphs)</span>
          <textarea id={id} className="input" rows={6} maxLength={4000} value={note} onChange={(e) => { setNote(e.target.value); setResult(null); }} />
        </label>
        <div className="row-actions">
          <button type="button" className="btn btn-sm" onClick={doPreview} disabled={look.busy}>Preview</button>
          <button type="button" className="btn btn-primary btn-sm" onClick={() => setConfirming(true)}
            disabled={!ready || !preview || preview.already_sent || send.busy}>Send to founders</button>
          {!ready && <span className="cap">Write at least a sentence, then preview.</span>}
        </div>
        {preview?.already_sent && <p className="cap" role="status">This month&apos;s briefing was sent{preview.sent_at ? ` ${etTime(preview.sent_at)}` : ""}.</p>}
        {result && <p className="notice" role="status">{result}</p>}
        <ErrorLine error={look.error ?? send.error} />
        {preview && (
          <div className="stack-xs">
            <p className="cap">Subject: <span className="strong">{preview.subject}</span> (as the first founder would see it)</p>
            <iframe title="Founder Briefing preview" className="briefing-frame" sandbox="" srcDoc={preview.html} />
          </div>
        )}
        {preview && preview.history.length > 0 && (
          <ul className="stack-xs" aria-label="Briefings sent">{preview.history.map((h) => (
            <li key={h.month} className="cap">{h.month}: sent {h.sent ?? 0}{h.failed ? `, failed ${h.failed}` : ""} by {h.sent_by}</li>
          ))}</ul>
        )}
      </div>
      <ConfirmDialog open={confirming} title="Send the Founder Briefing?" confirmLabel="Send now" busy={send.busy}
        error={<ErrorLine error={send.error} />} onConfirm={doSend} onClose={() => setConfirming(false)}
        body={<p className="body-sm">This emails {preview?.recipients ?? 0} active founders now. It can only be sent once this month.</p>} />
    </Card>
  );
}
