import type { Metadata } from "next";

import { WeeklyRecapView } from "@/features/WeeklyRecap";
import { publicMetadata } from "@/lib/seo";
import { loadRecapIndex } from "@/server/publicRecap";

export const dynamic = "force-dynamic";

export const metadata: Metadata = publicMetadata("/weekly", "Weekly recap: what HSF's scanner ranked",
  "Every week, the stocks HSF's market scanner ranked most and the biggest earnings coming up. A record of rankings, not recommendations.");

export default async function Page() {
  const { latest, weeks } = await loadRecapIndex();
  return <WeeklyRecapView recap={latest ?? null} weeks={weeks} />;
}
