import type { MetadataRoute } from "next";

import { PRIVATE_PATHS, siteOrigin } from "@/lib/seo";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: [...PRIVATE_PATHS] },
    sitemap: `${siteOrigin()}/sitemap.xml`,
  };
}
