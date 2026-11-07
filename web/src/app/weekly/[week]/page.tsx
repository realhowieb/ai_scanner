import type { Metadata } from "next";

import { JsonLd } from "@/components/JsonLd";
import { WeeklyRecapView, weekRange } from "@/features/WeeklyRecap";
import { OG_IMAGE } from "@/lib/seo";
import { articleSchema } from "@/lib/schema";
import { loadRecap, loadRecapIndex } from "@/server/publicRecap";

type Props = { params: Promise<{ week: string }> };

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const recap = await loadRecap((await params).week);
  if (!recap) return { title: "Weekly recap not found" };
  const title = `HSF weekly recap: ${weekRange(recap.start, recap.end)}`;
  const top = recap.ranked.slice(0, 3).map((r) => r.ticker).join(", ");
  const description = `The stocks HSF's scanner ranked most the week of ${weekRange(recap.start, recap.end)}${top ? `, led by ${top}` : ""}. A record of rankings, not recommendations.`;
  return {
    title: { absolute: `${title} · HSFinest.com` }, description,
    alternates: { canonical: `/weekly/${recap.week}` }, robots: { index: true, follow: true },
    openGraph: { type: "article", siteName: "HSFinest.com", title, description, url: `/weekly/${recap.week}`, images: [OG_IMAGE] },
    twitter: { card: "summary_large_image", title, description, images: [OG_IMAGE] },
  };
}

export default async function Page({ params }: Props) {
  const [recap, index] = await Promise.all([loadRecap((await params).week), loadRecapIndex()]);
  return (
    <>
      {recap && (
        <JsonLd data={articleSchema({
          headline: `HSF weekly recap: ${weekRange(recap.start, recap.end)}`,
          description: `The stocks HSF's scanner ranked most the week of ${weekRange(recap.start, recap.end)}. A record of rankings, not recommendations.`,
          path: `/weekly/${recap.week}`,
          datePublished: recap.built_at,
        })} />
      )}
      <WeeklyRecapView recap={recap} weeks={index.weeks} />
    </>
  );
}
