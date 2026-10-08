import type { Metadata } from "next";
import { Suspense } from "react";

import { ResetPasswordForm } from "@/features/AuthForms";

export const metadata: Metadata = { title: "Set a new password" };

export default function Page() {
  return (
    <main className="center">
      <Suspense>
        <ResetPasswordForm />
      </Suspense>
    </main>
  );
}
