import type { Metadata } from "next";
import { Suspense } from "react";

import { PaperView } from "@/features/PaperView";

export const metadata: Metadata = { title: "Paper trading" };

export default function Page() {
  return <Suspense><PaperView /></Suspense>;
}
