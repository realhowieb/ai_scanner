import type { Metadata } from "next";

import { publicMetadata } from "@/lib/seo";

import { PricingPage } from "@/features/Landing";

export const metadata: Metadata = publicMetadata("/pricing", "HSF AI plans and pricing",
  "Start free on Starter. Pro adds custom scans, email alerts and Day Trader; Premium adds AI research and paper trading. Cancel any time.");

export default function Page() {
  return <PricingPage />;
}
