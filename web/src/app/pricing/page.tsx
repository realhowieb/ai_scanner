import type { Metadata } from "next";

import { PricingPage } from "@/features/Landing";

export const metadata: Metadata = { title: "Plans and pricing" };

export default function Page() {
  return <PricingPage />;
}
