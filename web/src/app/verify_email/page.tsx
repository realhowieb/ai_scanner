import type { Metadata } from "next";
import { Suspense } from "react";

import { VerifyEmail } from "@/features/AuthForms";

export const metadata: Metadata = { title: "Verify email" };

export default function Page() {
  return (
    <main className="center">
      <Suspense>
        <VerifyEmail />
      </Suspense>
    </main>
  );
}
