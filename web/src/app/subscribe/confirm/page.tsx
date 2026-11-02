import type { Metadata } from "next";
import { Suspense } from "react";

import { ListConfirmView } from "@/features/EmailSignup";

export const metadata: Metadata = { title: "Weekly setups email" };

export default function Page() {
  return (
    <main className="center">
      <Suspense>
        <ListConfirmView />
      </Suspense>
    </main>
  );
}
