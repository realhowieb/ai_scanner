import type { Metadata } from "next";

import { JsonLd } from "@/components/JsonLd";
import { Landing } from "@/features/Landing";
import { publicMetadata } from "@/lib/seo";
import { organizationSchema, softwareApplicationSchema, webSiteSchema } from "@/lib/schema";

export const metadata: Metadata = publicMetadata("/", "HSF AI: Find Market Opportunities with AI-Powered Intelligence",
  "Discover momentum, analyze technical signals and monitor market activity. HSF AI scans thousands of U.S. stocks and explains the evidence behind every setup. Free Starter plan.");

// Signed-in visitors never see this: the proxy sends "/" to /today for them.
export default function Home() {
  return (
    <>
      <JsonLd data={[organizationSchema(), webSiteSchema(), softwareApplicationSchema(
        "Stock-market research platform: scheduled scans of thousands of U.S. stocks, ranked by the 0-100 HSF Score, with the technical evidence behind every setup. Free Starter plan.")]} />
      <Landing />
    </>
  );
}
