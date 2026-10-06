import type { Metadata } from "next";
import { Suspense } from "react";

import { WatchlistsView } from "@/features/WatchlistsView";

export const metadata: Metadata = { title: "Watchlists" };

export default function WatchlistsPage() {
  return (
    <Suspense>
      <WatchlistsView />
    </Suspense>
  );
}
