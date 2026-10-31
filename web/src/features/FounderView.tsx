"use client";

// Founding 100 for a signed-in account: the two tiers with seats left, the disclosures
// checkout must show, the acknowledgment (never pre-ticked) and the buy buttons. Members
// see their badge, number and end date, and the Pro-to-Elite upgrade.
import { useSearchParams } from "next/navigation";
import { useEffect, useId, useState } from "react";

import { founding } from "@/api/founding";
import type { FounderTier, FoundingOffer } from "@/api/founding";
import Link from "@/components/Link";
import { Card, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etDate } from "@/lib/format";

import { FounderBenefits } from "./FounderBenefits";
import { FounderRefunds } from "./FounderRefunds";
import { POLICY_LINKS } from "./PolicyPage";

type Disclosure = FoundingOffer["tiers"][number]["disclosure"];

function DisclosureList({ d }: { d: Disclosure }) {
  return (
    <div className="stack-xs">
      <p className="strong">{d.name}</p>
      <dl className="kv founder-disclosure" aria-label={`${d.name} payment terms`}>
        {[d.payment, d.duration, d.recurring, d.refund].map((line) => {
          const [k, ...v] = line.split(": ");
          return <div key={k}><dt>{k}</dt><dd className="strong">{v.join(": ")}</dd></div>;
        })}
      </dl>
    </div>
  );
}

function Acknowledge({ offer, checked, onChange }: { offer: FoundingOffer; checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId();
  return (
    <div className="stack-xs">
      <label className="check" htmlFor={id}>
        <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span>{offer.acknowledgment}</span>
      </label>
      <p className="cap">
        Read them first: {POLICY_LINKS.map((l, i) => (
          <span key={l.href}>{i > 0 && " · "}<Link href={l.href} target="_blank" rel="noopener">{l.label}<span className="sr-only"> (opens in a new tab)</span></Link></span>
        ))}
      </p>
    </div>
  );
}

export function FounderView() {
  const params = useSearchParams();
  const state = useApi("founding-offer", (signal) => founding.offer(signal));
  const buy = useAction();
  const [agreed, setAgreed] = useState(false);
  const back = params.get("checkout");
  const offer = state.data;
  const reload = state.reload;

  useEffect(() => {
    if (back === "success") {
      // The webhook may land a moment after Stripe sends the buyer back.
      const t = setTimeout(reload, 4000);
      return () => clearTimeout(t);
    }
  }, [back, reload]);

  if (state.error && !offer) {
    if (state.error.status === 404) {
      return <main id="main" className="page stack"><Card title="Founding 100"><p className="body-sm">The Founding 100 isn&apos;t open yet.</p></Card></main>;
    }
    return <main id="main" className="page"><ErrorState error={state.error} onRetry={reload} what="the Founding 100" /></main>;
  }
  if (!offer) return <main id="main" className="page"><Skeleton rows={4} label="Loading the Founding 100" /></main>;

  async function start(tier: FounderTier, upgrade = false) {
    const out = await buy.run(() => founding.checkout(offer!, tier, agreed, upgrade));
    if (out?.checkout_url) window.location.assign(out.checkout_url);
  }

  const m = offer.membership;
  const live = m && (m.status === "active" || m.status === "pending");
  return (
    <main id="main" className="page stack">
      <Link href="/account" className="back">← Back to Account</Link>
      <header className="stack-xs">
        <h1 className="h1">Founding 100</h1>
        <p className="body-sm">100 memberships, ever: 75 Founder Pro and 25 Founder Elite. One payment covers five calendar years. Nothing renews or charges again.</p>
      </header>
      {offer.preview && <p className="banner" role="note">Admin preview: the Founding 100 is still switched off for members.{offer.missing_details.length ? ` Still to set: ${offer.missing_details.join(", ")}.` : ""}</p>}
      {back === "success" && !live && <p className="banner" role="status">Payment received. Your membership appears here as soon as Stripe confirms it.</p>}
      {back === "cancelled" && <p className="cap" role="status">Checkout closed. Your seat is held for a few more minutes if you want to try again.</p>}

      {m && (
        <Card title="Your membership" aside={m.badge ? <Pill tone="gold">{m.badge}</Pill> : undefined}>
          <dl className="kv">
            <div><dt>Tier</dt><dd>{m.label}</dd></div>
            <div><dt>Status</dt><dd>{m.status}</dd></div>
            {m.activated_at && <div><dt>Started</dt><dd>{etDate(m.activated_at)}</dd></div>}
            {m.expires_at && <div><dt>Access until</dt><dd>{etDate(m.expires_at)}</dd></div>}
          </dl>
        </Card>
      )}
      {m && (m.status === "active" || m.status === "expired") && <FounderBenefits />}
      {m && <FounderRefunds />}

      {(!live || offer.upgrade.eligible) && (
        <Card title="Before you pay">
          <div className="stack-sm">
            <p className="body-sm">Tick this box first. The payment buttons stay off until you do.</p>
            <Acknowledge offer={offer} checked={agreed} onChange={setAgreed} />
            <ErrorLine error={buy.error} />
            {buy.error?.code === "policy_changed" && (
              <button type="button" className="btn btn-sm" onClick={() => { buy.clear(); setAgreed(false); reload(); }}>Load the current policies</button>
            )}
          </div>
        </Card>
      )}
      {!live && (
        <div className="founder-grid">
          {offer.tiers.map((t) => (
            <Card key={t.tier} title={t.label} aside={<span className="cap">{t.remaining} of {t.capacity} left</span>}>
              <div className="stack-sm">
                <p className="body-sm">Includes everything in {t.plan === "premium" ? "Premium" : "Pro"} for five years.{t.tier === "founder_elite" ? " Personal API access is planned, not available yet." : ""}</p>
                <DisclosureList d={t.disclosure} />
                <button type="button" className="btn btn-primary btn-block" disabled={!t.on_sale || !agreed || buy.busy}
                  onClick={() => start(t.tier)}>
                  {t.remaining === 0 ? "Sold out" : !offer.checkout_open ? "Not on sale yet" : `Continue to payment: ${t.disclosure.payment.replace("One-time payment: ", "")}`}
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {live && offer.upgrade.eligible && (
        <Card title="Upgrade to Founder Elite">
          <div className="stack-sm">
            <p className="body-sm">Keeps your founder number and your end date.</p>
            <DisclosureList d={offer.upgrade.disclosure} />
            <button type="button" className="btn btn-primary" disabled={!offer.upgrade.on_sale || !agreed || buy.busy}
              onClick={() => start("founder_elite", true)}>
              {offer.upgrade.on_sale ? "Continue to payment: $500" : "Upgrade not on sale"}
            </button>
          </div>
        </Card>
      )}

      <p className="cap">Payments are handled by Stripe. HSF AI never sees your card number. Not financial advice.</p>
    </main>
  );
}
