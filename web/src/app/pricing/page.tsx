import type { Metadata } from "next";

import { JsonLd } from "@/components/JsonLd";
import { publicMetadata } from "@/lib/seo";
import { softwareApplicationSchema } from "@/lib/schema";

import { PricingPage } from "@/features/Landing";

export const metadata: Metadata = publicMetadata("/pricing", "HSF AI plans and pricing",
  "Start free on Starter. Pro adds custom scans, email alerts and Day Trader; Premium adds AI research and paper trading. Cancel any time.");

export default function Page() {
  return (
    <>
      <JsonLd data={softwareApplicationSchema(
        "HSF AI stock-market research plans: free Starter, Pro and Premium. Cancel any time.")} />
      <PricingPage />
    </>
  );
}
