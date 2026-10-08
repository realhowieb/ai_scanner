import type { Metadata } from "next";

import { ScanHistoryView } from "@/features/ScanHistoryView";

export const metadata: Metadata = { title: "Scan history" };

export default function ScanHistoryPage() {
  return <ScanHistoryView />;
}
