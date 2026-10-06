import type { Metadata } from "next";
import { Suspense } from "react";

import { ScannerView } from "@/features/ScannerView";

export const metadata: Metadata = { title: "Scanner" };

export default function ScannerPage() {
  return (
    <Suspense>
      <ScannerView />
    </Suspense>
  );
}
