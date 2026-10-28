import type { Metadata } from "next";
import { Suspense } from "react";

import { FounderView } from "@/features/FounderView";

export const metadata: Metadata = { title: "Founding 100" };

export default function FounderPage() {
  return <Suspense><FounderView /></Suspense>;
}
