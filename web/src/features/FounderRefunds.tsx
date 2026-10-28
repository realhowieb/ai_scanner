"use client";

// A founding member's refunds: each payment with its 14-day guarantee date, the requests
// made and a Request refund dialog. The API picks the payment from the signed-in account;
// the page only says which kind (purchase or upgrade).
import { useState } from "react";

import { cents, REFUND_STATUS_LABEL, refunds } from "@/api/founding";
import type { FounderPayment, MyRefunds, RefundRequest } from "@/api/founding";
import { Dialog } from "@/components/Dialog";
import Link from "@/components/Link";
import { Card, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etDate } from "@/lib/format";

const KIND_LABEL = { purchase: "Membership payment", upgrade: "Upgrade payment" } as const;
const OPEN = new Set(["requested", "approved", "initiated"]);
const TONE = { requested: "warn", approved: "up", initiated: "up", completed: "up", denied: "down", failed: "down" } as const;

function Ask({ payment, onClose, onSent }: { payment: FounderPayment; onClose: () => void; onSent: () => void }) {
  const [reason, setReason] = useState("");
  const act = useAction();
  const send = () => void act.run(async () => {
    await refunds.request(payment.kind, reason);
    onSent();
    onClose();
    return true;
  });
  return (
    <Dialog open title={`Request a refund: ${KIND_LABEL[payment.kind].toLowerCase()}`} onClose={onClose}
      footer={<>
        <button type="button" className="btn" onClick={onClose}>Cancel</button>
        <button type="button" className="btn btn-primary" onClick={send} disabled={act.busy}>{act.busy ? "Sending…" : "Send refund request"}</button>
      </>}>
      <div className="stack-sm">
        {payment.within_guarantee
          ? <p className="body-sm">You&apos;re inside the 14-day guarantee for this payment, so you&apos;ll get a full refund of {cents(payment.refundable_cents)}. You don&apos;t need to give a reason.</p>
          : <p className="body-sm">The 14-day guarantee for this payment ended on {etDate(payment.deadline)}. You can still ask: a person reviews it against the refund policy and replies.</p>}
        <p className="cap">
          {payment.kind === "upgrade"
            ? "Refunding the upgrade returns you to Founder Pro with the same number and end date."
            : "A full refund of this payment ends the founding membership. Your HSF AI account stays."}
          {" "}Nothing changes while we review it.
        </p>
        <label className="field"><span>Anything we should know? (optional)</span>
          <textarea rows={3} maxLength={1000} value={reason} onChange={(e) => setReason(e.target.value)} data-autofocus />
        </label>
        <ErrorLine error={act.error} />
      </div>
    </Dialog>
  );
}

function RequestRow({ r }: { r: RefundRequest }) {
  return (
    <li className="stack-xs">
      <div className="row-actions"><strong>{KIND_LABEL[r.kind]}</strong><Pill tone={TONE[r.status]}>{REFUND_STATUS_LABEL[r.status]}</Pill></div>
      <p className="cap">
        Asked {etDate(r.requested_at)}{r.within_guarantee ? " (within 14 days)" : ""}
        {r.status === "requested" ? ` · we'll reply by ${etDate(r.due_at)}` : ""}
        {r.amount_cents && r.status !== "denied" ? ` · ${cents(r.amount_cents)}` : ""}
        {r.status === "initiated" ? " · sent to your card; when it shows depends on your bank" : ""}
      </p>
      {r.decision_reason && <p className="cap">Reason: {r.decision_reason}</p>}
    </li>
  );
}

export function RefundsBody({ data, onChanged }: { data: MyRefunds; onChanged: () => void }) {
  const [asking, setAsking] = useState<FounderPayment | null>(null);
  const openKinds = new Set(data.requests.filter((r) => OPEN.has(r.status)).map((r) => r.kind));
  return (
    <div className="stack-sm">
      <ul className="stack-sm">
        {data.payments.map((p) => (
          <li key={p.kind} className="stack-xs">
            <div className="row-actions">
              <strong>{KIND_LABEL[p.kind]}: {cents(p.amount_cents)}</strong>
              <span className="cap">paid {etDate(p.paid_at)}</span>
            </div>
            <p className="cap">
              {p.refundable_cents <= 0 ? "Refunded in full."
                : p.within_guarantee ? `Full refund guarantee until ${etDate(p.deadline)}.`
                  : `14-day guarantee ended ${etDate(p.deadline)}. You can still ask; a person reviews it.`}
              {p.refunded_cents > 0 && p.refundable_cents > 0 ? ` ${cents(p.refunded_cents)} refunded so far.` : ""}
            </p>
            {p.refundable_cents > 0 && (
              openKinds.has(p.kind)
                ? <p className="cap">Your request for this payment is open.</p>
                : <div><button type="button" className="btn btn-sm" onClick={() => setAsking(p)}>Request refund</button></div>
            )}
          </li>
        ))}
      </ul>
      {data.requests.length > 0 && (
        <>
          <h3 className="h3">Your requests</h3>
          <ul className="stack-sm">{[...data.requests].reverse().map((r) => <RequestRow key={r.id} r={r} />)}</ul>
        </>
      )}
      <p className="cap">
        Approved refunds are started within {data.start_within_business_days} business days and go back to your original payment method.
        {" "}Read the <Link href="/founding-100/refunds">Refund Policy</Link>. Questions: {data.support_email}.
      </p>
      {asking && <Ask payment={asking} onClose={() => setAsking(null)} onSent={onChanged} />}
    </div>
  );
}

export function FounderRefunds() {
  const state = useApi("founding-refunds", (signal) => refunds.mine(signal));
  if (state.error && !state.data) return <ErrorState error={state.error} onRetry={state.reload} what="your refunds" />;
  if (!state.data) return <Skeleton rows={2} label="Loading your payments" />;
  if (state.data.payments.length === 0) return null;
  return <Card title="Payments and refunds"><RefundsBody data={state.data} onChanged={state.reload} /></Card>;
}
