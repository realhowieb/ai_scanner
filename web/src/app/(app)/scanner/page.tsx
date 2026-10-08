import type { Metadata } from "next";
import { Suspense } from "react";

import { PriceTape } from "@/features/PriceTape";
import { ScannerView } from "@/features/ScannerView";

export const metadata: Metadata = { title: "Scanner" };

export default function ScannerPage() {
  return (
    <div className="stack">
      <PriceTape />
      <Suspense>
        <ScannerView />
      </Suspense>
    </div>
  );
}
