import type { MetadataRoute } from "next";

import { PUBLIC_PAGES, siteOrigin } from "@/lib/seo";
import { loadRecapIndex } from "@/server/publicRecap";

export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const origin = siteOrigin();
  const pages: MetadataRoute.Sitemap = PUBLIC_PAGES.map((path) => ({
    url: `${origin}${path === "/" ? "" : path}`,
    changeFrequency: path === "/" || path === "/weekly" ? "daily" : "weekly",
    priority: path === "/" ? 1 : 0.6,
  }));
  const { weeks } = await loadRecapIndex();
  return [...pages, ...weeks.map((w) => ({ url: `${origin}/weekly/${w.week}`, changeFrequency: "yearly" as const, priority: 0.4 }))];
}
