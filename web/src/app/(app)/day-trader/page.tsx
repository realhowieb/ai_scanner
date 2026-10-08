import type { Metadata } from "next";

import { DayTraderView } from "@/features/DayTraderView";

export const metadata: Metadata = { title: "Day Trader" };

export default function Page() {
  return <DayTraderView />;
}
