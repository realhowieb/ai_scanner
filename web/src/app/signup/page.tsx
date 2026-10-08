import type { Metadata } from "next";
import { Suspense } from "react";

import { SignupForm } from "@/features/AuthForms";

export const metadata: Metadata = { title: "Create account" };

export default function Page() {
  return (
    <main className="center">
      <Suspense>
        <SignupForm />
      </Suspense>
    </main>
  );
}
