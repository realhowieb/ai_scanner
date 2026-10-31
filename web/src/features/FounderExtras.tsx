"use client";

// Founder page extras: the value meter, the shareable founder card and the opt-in
// Founders Wall name. All read from GET /v1/founding/benefits.
import { useId, useState } from "react";

import { cents, foundersWall } from "@/api/founding";
import type { FounderBenefits as Benefits } from "@/api/founding";
import Link from "@/components/Link";
import { Card, ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { inviteUrl } from "./InviteFriend";

const whole = (c: number) => cents(c).replace(/\.00$/, "");

/** Months used of 60, what they'd have cost on the monthly plan, and the payback month. */
export function ValueMeter({ data }: { data: Benefits }) {
  const v = data.value;
  if (!v) return null;
  const pct = Math.round((v.months_used / v.term_months) * 100);
  return (
    <Card title="Your founder value">
      <div className="stack-sm">
        <div className="meter" role="meter" aria-label="Months of your membership used" aria-valuemin={0}
          aria-valuemax={v.term_months} aria-valuenow={v.months_used}>
          <span style={{ width: `${Math.max(pct, 1)}%` }} />
        </div>
        <dl className="kv">
          <div><dt>Months used</dt><dd className="strong">{v.months_used} of {v.term_months}</dd></div>
          <div><dt>Months left</dt><dd>{v.months_left}</dd></div>
          <div><dt>Paying monthly so far</dt><dd>{whole(v.monthly_cost_so_far_cents)} <span className="cap">({whole(v.monthly_cents)} × {v.months_used})</span></dd></div>
          <div><dt>You paid once</dt><dd>{whole(v.paid_cents)}</dd></div>
          {v.alerts_used != null && data.elite_limits && (
            <div><dt>Alerts in use</dt><dd>{v.alerts_used} of {data.elite_limits.alert_limit}</dd></div>
          )}
        </dl>
        <p className="body-sm strong">
          {v.ahead_cents > 0 ? `You're ${whole(v.ahead_cents)} ahead of paying monthly.` : `Your seat pays for itself in month ${v.break_even_month}.`}
        </p>
      </div>
    </Card>
  );
}

/** A link to the member's public founder card (number, tier, dates; never a name or email). */
export function ShareCard({ data }: { data: Benefits }) {
  const [copied, setCopied] = useState<"ok" | "fail" | null>(null);
  if (!data.card_path) return null;
  const url = inviteUrl(data.card_path);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied("ok");
    } catch {
      setCopied("fail");
    }
  };
  const text = encodeURIComponent(`I'm ${data.badge ?? "an HSF AI founder"}. One of 100 founding members of HSFinest.AI.`);
  return (
    <Card title="Share your founder card">
      <div className="stack-sm">
        <p className="body-sm">Your founder number, tier and dates as a card. Friends who join from it use your invite link.
          {data.show_badge ? "" : " Turn on \"Show my founder badge\" below to share it."}</p>
        {data.show_badge && (
          <>
            <div className="invite-link">
              <input readOnly value={url} aria-label="Your founder card link" onFocus={(e) => e.currentTarget.select()} />
              <button type="button" className="btn btn-sm" onClick={copy}>Copy link</button>
            </div>
            <div className="row-actions">
              <a className="btn btn-sm" href={`https://twitter.com/intent/tweet?text=${text}&url=${encodeURIComponent(url)}`}
                target="_blank" rel="noopener noreferrer">Post to X<span className="sr-only"> (opens in a new tab)</span></a>
              <Link href={data.card_path} className="cap">Preview your card</Link>
            </div>
            {copied && <p className="cap" role="status">{copied === "ok" ? "Link copied." : "Couldn't copy. Select the link and copy it."}</p>}
          </>
        )}
      </div>
    </Card>
  );
}

/** Opt in to the public Founders Wall under a display name. Off by default. */
export function WallSetting({ data, onChange }: { data: Benefits; onChange: (b: Benefits) => void }) {
  const act = useAction();
  const id = useId();
  const listed = !!data.wall_name;
  const [on, setOn] = useState(listed);
  const [name, setName] = useState(data.wall_name ?? "");
  const save = (show: boolean) => void act.run(async () => {
    onChange(await foundersWall.set(show, show ? name.trim() : undefined));
    return true;
  });
  return (
    <fieldset className="stack-xs">
      <legend className="strong">Founders Wall</legend>
      <label className="check">
        <input type="checkbox" checked={on} disabled={act.busy}
          onChange={(e) => { setOn(e.target.checked); if (!e.target.checked && listed) save(false); }} />
        <span>List me on the public <Link href="/founders">Founders Wall</Link> under a display name I choose.
          Anyone can see it next to my founder number. Leave this off to show only &quot;Founder&quot;.</span>
      </label>
      {on && (
        <div className="row-actions">
          <label className="sr-only" htmlFor={id}>Display name</label>
          <input id={id} className="input" maxLength={30} value={name} placeholder="Display name or handle"
            onChange={(e) => setName(e.target.value)} />
          <button type="button" className="btn btn-sm" disabled={act.busy || name.trim().length < 2 || name.trim() === data.wall_name}
            onClick={() => save(true)}>{listed ? "Update name" : "Add me to the wall"}</button>
        </div>
      )}
      {listed && <p className="cap" role="status">Listed as &quot;{data.wall_name}&quot;.</p>}
      <p className="cap">2 to 30 letters or numbers. No email addresses or links. You can take it off any time.</p>
      <ErrorLine error={act.error} />
    </fieldset>
  );
}
