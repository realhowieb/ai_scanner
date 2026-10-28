// Search and link-preview metadata for the public marketing pages. Every other page keeps
// the root layout's noindex, so signed-in pages and the API never get listed.
import type { Metadata } from "next";

/** The pages search engines may list (also the sitemap). */
export const PUBLIC_PAGES = ["/", "/pricing", "/how-hsf-works", "/demo"] as const;
export type PublicPage = (typeof PUBLIC_PAGES)[number];

/** Signed-in areas and server routes that crawlers are asked to skip (robots.txt). */
export const PRIVATE_PATHS = ["/api/", "/today", "/scanner", "/stocks", "/watchlists", "/alerts", "/account",
  "/track-record", "/brief", "/journal", "/ask", "/day-trader", "/paper", "/admin", "/billing"] as const;

export function siteOrigin(env: Record<string, string | undefined> = process.env): string {
  return (env.NEXT_PUBLIC_SITE_URL || "https://app.hsfinest.com").replace(/\/+$/, "");
}

/** Title, description, canonical URL, Open Graph and X card for one public page. */
export function publicMetadata(path: PublicPage, title: string, description: string): Metadata {
  return {
    title: { absolute: `${title} · HSFinest.AI` },
    description,
    alternates: { canonical: path },
    robots: { index: true, follow: true },
    openGraph: { type: "website", siteName: "HSFinest.AI", title, description, url: path },
    twitter: { card: "summary_large_image", title, description },
  };
}
