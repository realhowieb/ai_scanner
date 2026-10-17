import type { Metadata } from "next";
import { Suspense } from "react";

import { CompareView } from "@/features/CompareView";

export const metadata: Metadata = { title: "Compare stocks" };

export default function ComparePage() {
  return (
    <Suspense>
      <CompareView />
    </Suspense>
  );
}
