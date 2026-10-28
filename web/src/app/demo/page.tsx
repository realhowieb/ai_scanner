import type { Metadata } from "next";

import { publicMetadata } from "@/lib/seo";

import { DemoPage } from "@/features/Demo";

export const metadata: Metadata = publicMetadata("/demo", "HSF AI interactive demo",
  "Explore HSF AI with sample scanner, watchlist, alert, stock detail and AI research data. No account needed.");

export default function Page() {
  return <DemoPage />;
}
