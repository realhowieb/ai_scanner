import type { Metadata } from "next";

import { BriefView } from "@/features/BriefView";

export const metadata: Metadata = { title: "Market Brief" };

export default function BriefPage() {
  return <BriefView />;
}
