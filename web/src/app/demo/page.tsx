import type { Metadata } from "next";

import { DemoPage } from "@/features/Demo";

export const metadata: Metadata = {
  title: "Demo",
  description: "Explore HSFinest.AI with polished sample scanner, watchlist, alert, stock detail, and AI research data.",
};

export default function Page() {
  return <DemoPage />;
}
