import type { Metadata } from "next";

import { CustomScanView } from "@/features/CustomScanView";

export const metadata: Metadata = { title: "Custom scan" };

export default function CustomScanPage() {
  return <CustomScanView />;
}
