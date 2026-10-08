import type { Metadata } from "next";
import { Suspense } from "react";

import { UnsubscribeView } from "@/features/UnsubscribeView";

export const metadata: Metadata = { title: "Email preferences" };

export default function Page() {
  return (
    <main className="center">
      <Suspense>
        <UnsubscribeView />
      </Suspense>
    </main>
  );
}
