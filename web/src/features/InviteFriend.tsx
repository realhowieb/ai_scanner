"use client";

// Invite a friend: your share link (/signup?ref=CODE), how many joined with it, and the
// reward. Each friend who signs up with the link and confirms their email gives both of
// you free Pro for `reward_days` (the API grants it; billing is untouched). Hidden when
// the API has no such route or invites are switched off.
import { useState } from "react";

import { insights } from "@/api/userData";
import { Card } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etDate } from "@/lib/format";

export function inviteUrl(path: string, origin = typeof window === "undefined" ? "" : window.location.origin): string {
  return `${origin}${path}`;
}

export function InviteFriendCard() {
  const res = useApi("referrals", (signal) => insights.referrals(signal));
  const [copied, setCopied] = useState<"ok" | "fail" | null>(null);
  const d = res.data;
  if (res.error || !d || typeof d.code !== "string") return null;
  const url = inviteUrl(d.path);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied("ok");
    } catch {
      setCopied("fail");
    }
  };
  const share = typeof navigator !== "undefined" && "share" in navigator;
  return (
    <Card title="Invite a friend" id="invite">
      <p className="body-sm">
        {d.reward_days > 0
          ? <>Share your link. When a friend signs up with it and confirms their email, you both get <b>{d.reward_days} days of Pro</b> free.</>
          : <>Share your link with friends who trade.</>}
      </p>
      <div className="invite-link">
        <input readOnly value={url} aria-label="Your invite link" onFocus={(e) => e.currentTarget.select()} />
        <button type="button" className="btn" onClick={() => void copy()}>{copied === "ok" ? "Copied" : "Copy link"}</button>
        {share && (
          <button type="button" className="btn" onClick={() => void navigator.share({ title: "HSFinest.AI", text: "Stock setups ranked by HSF Score. Join with my link:", url }).catch(() => undefined)}>
            Share
          </button>
        )}
      </div>
      {copied === "fail" && <p className="cap" role="alert">Couldn&apos;t copy. Select the link and copy it yourself.</p>}
      <div className="metric-strip" aria-label="Invite results">
        <span><strong>{d.signed_up}</strong> signed up</span>
        <span><strong>{d.verified}</strong> confirmed</span>
        {d.pro_trial_until && <span>Free Pro until <strong>{etDate(d.pro_trial_until)}</strong></span>}
      </div>
      <p className="cap">Free Pro time adds up for each friend, up to six months ahead. It never charges you and ends on its own.</p>
    </Card>
  );
}
