import { describe, expect, it } from "vitest";

import robots from "@/app/robots";
import sitemap from "@/app/sitemap";
import { PRIVATE_PATHS, PUBLIC_PAGES, publicMetadata, siteOrigin } from "@/lib/seo";

describe("search metadata", () => {
  it("lists only public pages in the sitemap", () => {
    const urls = sitemap().map((e) => e.url);
    expect(urls).toEqual(PUBLIC_PAGES.map((p) => `${siteOrigin()}${p === "/" ? "" : p}`));
    for (const u of urls) {
      expect(PRIVATE_PATHS.some((p) => new URL(u).pathname.startsWith(p))).toBe(false);
    }
  });

  it("keeps crawlers out of the API and signed-in pages", () => {
    const r = robots();
    const rule = Array.isArray(r.rules) ? r.rules[0]! : r.rules;
    expect(rule.disallow).toEqual(expect.arrayContaining(["/api/", "/today", "/account", "/admin", "/watchlists"]));
    expect(r.sitemap).toBe(`${siteOrigin()}/sitemap.xml`);
  });

  it("public pages are indexable with a canonical URL and preview cards", () => {
    const m = publicMetadata("/pricing", "Plans", "Desc");
    expect(m.robots).toEqual({ index: true, follow: true });
    expect(m.alternates?.canonical).toBe("/pricing");
    expect(m.openGraph).toMatchObject({ title: "Plans", url: "/pricing" });
    expect(m.twitter).toMatchObject({ card: "summary_large_image" });
  });

  it("uses the configured site address", () => {
    expect(siteOrigin({ NEXT_PUBLIC_SITE_URL: "https://example.test/" })).toBe("https://example.test");
    expect(siteOrigin({})).toBe("https://app.hsfinest.com");
  });
});
