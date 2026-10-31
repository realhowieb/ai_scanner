"use client";

// The Founding 100 offer on the landing and pricing pages and in the account menu, from
// GET /v1/public/founding. Shows nothing while the program is closed (or the call fails).
import Link from "@/components/Link";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { track } from "@/lib/funnel";
import { useOptionalMe } from "@/session/SessionProvider";

type PublicFounding = Schemas["PublicFounding"];

export const FOUNDER_HREF = "/account/founder";

export function usePublicFounding(): PublicFounding | null {
  const state = useApi<PublicFounding>("public-founding", (signal) => unwrap(publicApi.GET("/v1/public/founding", { signal })));
  return state.data?.open ? state.data : null;
}

const dollars = (c: number) => `$${Math.round(c / 100).toLocaleString("en-US")}`;
type Tier = PublicFounding["tiers"][number];
const PLAN_NAME: Record<string, string> = { founder_pro: "Pro", founder_elite: "Premium" };

/** "5 years of Pro: $1,500 paying monthly, $499 once." */
export function fiveYearLine(t: Tier): string | null {
  const f = t.five_year;
  if (!f || !f.monthly_cents) return null;
  return `5 years of ${PLAN_NAME[t.tier] ?? "HSF"}: ${dollars(f.five_years_monthly_cents)} paying monthly, ${dollars(t.price_cents)} once.`;
}

/** The 5-year comparison, one row per tier. */
export function FiveYearMath({ tiers }: { tiers: Tier[] }) {
  const rows = tiers.filter((t) => t.five_year?.monthly_cents);
  if (!rows.length) return null;
  return (
    <ul className="five-year" aria-label="Five years compared">
      {rows.map((t) => (
        <li key={t.tier}>
          <span className="strong">{t.label}</span>
          <span className="cap">5 years of {PLAN_NAME[t.tier]} paying monthly <s>{dollars(t.five_year!.five_years_monthly_cents)}</s></span>
          <span className="strong five-year-price">{dollars(t.price_cents)} once</span>
          <span className="cap gain">You keep {dollars(t.five_year!.saved_cents)}</span>
        </li>
      ))}
    </ul>
  );
}

/** A one-line Founding 100 nudge for upgrade prompts and the sidebar (free and Pro plans). */
export function FoundingNudge({ surface, compact = false }: { surface: string; compact?: boolean }) {
  const d = usePublicFounding();
  const me = useOptionalMe();
  if (!d || d.remaining <= 0 || me?.founder_tier) return null;
  const pro = d.tiers.find((t) => t.tier === "founder_pro");
  return (
    <Link href={FOUNDER_HREF} className={compact ? "founding-nudge cap" : "founding-nudge"}
      onClick={() => track("primary_cta_click", `founding_${surface}`)}>
      Founding 100: {d.remaining} seats left{!compact && pro ? `. Founder Pro is ${dollars(pro.price_cents)} once for 5 years` : ""} →
    </Link>
  );
}

export function FoundingBanner({ surface }: { surface: string }) {
  const d = usePublicFounding();
  if (!d) return null;
  const onSale = d.tiers.filter((t) => t.on_sale);
  const soldOut = d.remaining <= 0;
  return (
    <section aria-labelledby="founding-h" className="card stack-xs founding-banner">
      <p className="eyebrow">Founding 100</p>
      <h2 id="founding-h" className="h2">{soldOut ? "The Founding 100 is sold out" : `${d.remaining} of 100 founding seats left`}</h2>
      {!soldOut && (
        <p className="body-sm">
          One payment, five years of HSF. {d.tiers.map((t) => `${t.label} ${dollars(t.price_cents)} (${t.remaining} left)`).join(" · ")}.
          {" "}No renewals, and a 14-day refund guarantee.
        </p>
      )}
      {!soldOut && <FiveYearMath tiers={d.tiers} />}
      {!soldOut && (
        <div className="row-actions">
          <Link href={FOUNDER_HREF} className="btn btn-primary" onClick={() => track("primary_cta_click", `founding_${surface}`)}>
            {onSale.length ? "See the Founding 100" : "Learn about the Founding 100"}
          </Link>
          <Link href="/founders" className="cap">See the Founders Wall</Link>
          <Link href="/founding-100/terms" className="cap">Founding terms</Link>
        </div>
      )}
    </section>
  );
}
