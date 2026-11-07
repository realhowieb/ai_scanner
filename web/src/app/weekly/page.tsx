import type { Metadata } from "next";

import { JsonLd } from "@/components/JsonLd";
import { WeeklyRecapView } from "@/features/WeeklyRecap";
import { publicMetadata } from "@/lib/seo";
import { weekRange } from "@/lib/weekRange";
import { articleSchema } from "@/lib/schema";
import { loadRecapIndex } from "@/server/publicRecap";

export const dynamic = "force-dynamic";

export const metadata: Metadata = publicMetadata("/weekly", "Weekly recap: what HSF's scanner ranked",
  "Every week, the stocks HSF's market scanner ranked most and the biggest earnings coming up. A record of rankings, not recommendations.");

export default async function Page() {
  const { latest, weeks } = await loadRecapIndex();
  return (
    <>
      {latest && (
        <JsonLd data={articleSchema({
          headline: `HSF weekly recap: ${weekRange(latest.start, latest.end)}`,
          description: `The stocks HSF's scanner ranked most the week of ${weekRange(latest.start, latest.end)}. A record of rankings, not recommendations.`,
          path: `/weekly/${latest.week}`,
          datePublished: latest.built_at,
        })} />
      )}
      <WeeklyRecapView recap={latest ?? null} weeks={weeks} />
    </>
  );
}
