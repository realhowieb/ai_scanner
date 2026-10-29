"use client";

// What a founding member gets beyond their plan: the founder rate after five years,
// early access, founder invites, Elite limits, the badge on shared links and the poll.
import { useId, useState } from "react";

import { founderBenefits } from "@/api/founding";
import type { FounderBenefits as Benefits } from "@/api/founding";
import Link from "@/components/Link";
import { Card, ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";

function Poll({ data, onChange }: { data: Benefits; onChange: (b: Benefits) => void }) {
  const act = useAction();
  const name = useId();
  const [picked, setPicked] = useState<string | null>(data.poll.choice ?? null);
  if (!data.poll.options.length) return null;
  const save = () => void act.run(async () => {
    if (picked) onChange(await founderBenefits.vote(picked));
    return true;
  });
  return (
    <fieldset className="stack-xs" disabled={!data.poll.open || act.busy}>
      <legend className="strong">What should we build next?</legend>
      <p className="cap">Founder poll. One vote each; you can change it any time. Results guide the roadmap.</p>
      {data.poll.options.map((o) => (
        <label key={o.key} className="check">
          <input type="radio" name={name} value={o.key} checked={picked === o.key} onChange={() => setPicked(o.key)} />
          <span>{o.label}</span>
        </label>
      ))}
      <div className="row-actions">
        <button type="button" className="btn btn-sm" onClick={save} disabled={!picked || picked === data.poll.choice}>
          {data.poll.choice ? "Change my vote" : "Vote"}
        </button>
        {data.poll.choice && <span className="cap">Your vote: {data.poll.options.find((o) => o.key === data.poll.choice)?.label}</span>}
      </div>
      <ErrorLine error={act.error} />
    </fieldset>
  );
}

export function BenefitsBody({ data, onChange }: { data: Benefits; onChange: (b: Benefits) => void }) {
  const act = useAction();
  const toggle = (show: boolean) => void act.run(async () => {
    onChange(await founderBenefits.showBadge(show));
    return true;
  });
  return (
    <div className="stack-sm">
      <ul className="stack-xs">
        <li><strong>Founder rate for life.</strong> When your five years end, any HSF plan is {data.alumni_discount_percent}% off for as long as you keep it.</li>
        {data.elite_limits && (
          <li><strong>Elite limits.</strong> Up to {data.elite_limits.alert_limit} active alerts and {data.elite_limits.ai_multiplier}× the daily AI allowance.</li>
        )}
        {data.invites && (
          <li><strong>Founder invites.</strong> Friends who join with your invite link get {data.invites.days} days of Pro. {data.invites.left} of {data.invites.total} left. <Link href="/account#invite">Get your invite link</Link></li>
        )}
        {data.early_access.length > 0 && (
          <li><strong>Early access.</strong> {data.early_access.map((f) => f.label).join(", ")}.</li>
        )}
        {data.status === "active" && data.early_access.length === 0 && (
          <li><strong>Early access.</strong> New features open to founders first. We&apos;ll email you when one is ready.</li>
        )}
      </ul>
      <label className="check">
        <input type="checkbox" checked={data.show_badge} disabled={act.busy} onChange={(e) => toggle(e.target.checked)} />
        <span>Show my founder badge on stock pages I share (never your name or email)</span>
      </label>
      <ErrorLine error={act.error} />
      <Poll data={data} onChange={onChange} />
    </div>
  );
}

export function FounderBenefits() {
  const state = useApi("founding-benefits", (signal) => founderBenefits.get(signal));
  const [fresh, setFresh] = useState<Benefits | null>(null);
  const data = fresh ?? state.data;
  if (!data) return null;
  return <Card title="Founder benefits"><BenefitsBody data={data} onChange={setFresh} /></Card>;
}
