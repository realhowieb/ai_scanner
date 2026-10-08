import type { Metadata } from "next";

import { StockSearchView } from "@/features/StockSearchView";

export const metadata: Metadata = { title: "Stock Intelligence" };

export default function Page() {
  return <StockSearchView />;
}
