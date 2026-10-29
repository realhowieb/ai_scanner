"use client";

// The Founding 100 offer on the landing and pricing pages and in the account menu, from
// GET /v1/public/founding. Shows nothing while the program is closed (or the call fails).
import Link from "@/components/Link";

import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { track } from "@/lib/funnel";

type PublicFounding = Schemas["PublicFounding"];

export const FOUNDER_HREF = "/account/founder";

export function usePublicFounding(): PublicFounding | null {
  const state = useApi<PublicFounding>("public-founding", (signal) => unwrap(publicApi.GET("/v1/public/founding", { signal })));
  return state.data?.open ? state.data : null;
}

const dollars = (c: number) => `$${Math.round(c / 100).toLocaleString("en-US")}`;

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
      {!soldOut && (
        <div className="row-actions">
          <Link href={FOUNDER_HREF} className="btn btn-primary" onClick={() => track("primary_cta_click", `founding_${surface}`)}>
            {onSale.length ? "See the Founding 100" : "Learn about the Founding 100"}
          </Link>
          <Link href="/founding-100/terms" className="cap">Founding terms</Link>
        </div>
      )}
    </section>
  );
}
