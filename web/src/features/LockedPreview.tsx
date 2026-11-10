"use client";

// One real example behind a locked Pro feature (GET /v1/previews/{kind}): the top Day
// Trader mover, the biggest pre-market gap or the biggest company reporting this week,
// plus how many more the full feature shows. Starter sees what an upgrade adds before
// paying. Shows nothing while loading or when there's nothing to show.
import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";

type Kind = Schemas["FeaturePreview"]["kind"];

const LEAD: Record<Kind, string> = { day_trader: "Top mover right now", gappers: "Biggest pre-market gap", earnings: "Biggest report this week" };
const MORE: Record<Kind, (n: number) => string> = {
  day_trader: (n) => `${n} more movers ranked in Day Trader`,
  gappers: (n) => `${n} more stocks gapping`,
  earnings: (n) => `${n} more companies report this week`,
};

export function LockedPreview({ kind }: { kind: Kind }) {
  const { data } = useApi(`preview:${kind}`, (signal) => unwrap(api.GET("/v1/previews/{kind}", { params: { path: { kind } }, signal })));
  const ex = data?.example;
  if (!data || !ex) return null;
  const more = Math.max(0, data.count - 1);
  return (
    <p className="locked-preview" data-testid={`preview-${kind}`}>
      <span className="cap">{LEAD[kind]}: </span><TickerLink ticker={ex.ticker} logo={false} /> <span>{ex.detail}</span>
      {more > 0 && <span className="cap"> · {MORE[kind](more)}</span>}
    </p>
  );
}
