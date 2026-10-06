import type { Metadata } from "next";

import { AlertsView } from "@/features/AlertsView";

export const metadata: Metadata = { title: "Alerts" };

export default function AlertsPage() {
  return <AlertsView />;
}
