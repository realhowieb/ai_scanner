import type { Metadata } from "next";
import { Suspense } from "react";

import { SignupForm } from "@/features/AuthForms";
import { termsPublished } from "@/server/legalStatus";

export const metadata: Metadata = { title: "Create account" };

export const dynamic = "force-dynamic";

export default async function Page() {
  const terms = await termsPublished();
  return (
    <main className="center">
      <Suspense>
        <SignupForm terms={terms} />
      </Suspense>
    </main>
  );
}
