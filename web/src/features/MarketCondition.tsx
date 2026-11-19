"use client";

// The market condition badge (Scanner, Market Brief, stock pages): Trending up, Choppy or
// Risk-off from one transparent SPY rule (GET /v1/market/condition), with what it means for
// a setup. The same rule splits the Track record, so "choppy" means the same thing in both.
// Renders nothing when the condition can't be computed; it is never guessed.
import { publicApi, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Pill } from "@/components/ui";
import Link from "@/components/Link";
import { useApi } from "@/hooks/useApi";

export type Condition = Schemas["MarketCondition"];

export const CONDITION_TONE: Record<string, "up" | "warn" | "down"> = { trending_up: "up", choppy: "warn", risk_off: "down" };

export function useMarketCondition() {
  return useApi<Condition>("market-condition", (signal) => unwrap(publicApi.GET("/v1/market/condition", { signal })));
}

export function ConditionPill({ condition, label }: { condition?: string | null; label?: string | null }) {
  if (!condition || !label) return null;
  return <Pill tone={CONDITION_TONE[condition]}>{label}</Pill>;
}

/** `compact`: the pill and a one-line hint (stock pages); otherwise the advice too. */
export function MarketConditionBadge({ compact = false }: { compact?: boolean }) {
  const c = useMarketCondition().data;
  if (!c?.available || !c.condition) return null;
  return (
    <div className={`market-condition mc-${c.condition}`} role="note" aria-label={`Market condition: ${c.label}`}>
      <p className="mc-line"><span className="cap">Market</span> <ConditionPill condition={c.condition} label={c.label} />
        {compact && c.condition !== "trending_up" && <span className="cap"> Setups fail more often in this market.</span>}</p>
      {!compact && <p className="cap">{c.advice}</p>}
      <details className="cap"><summary>How this is decided</summary>
        <p>{c.why} Trending up: above both averages, the 20-day rising and volatility under 25%. Risk-off: below both averages or volatility 25% or more. Choppy: anything in between. Based on SPY&apos;s close on {c.as_of}. <Link href="/how-hsf-works#method-market">More on market conditions</Link></p>
      </details>
    </div>
  );
}
