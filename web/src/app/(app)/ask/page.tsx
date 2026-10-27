import type { Metadata } from "next";

import { AskHSFView } from "@/features/AskHSF";

export const metadata: Metadata = { title: "Ask HSF" };

export default function Page() {
  return <AskHSFView />;
}
