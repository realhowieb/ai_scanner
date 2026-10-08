import type { Metadata } from "next";
import { Suspense } from "react";

import { ForgotPasswordForm } from "@/features/AuthForms";

export const metadata: Metadata = { title: "Reset password" };

export default function Page() {
  return (
    <main className="center">
      <Suspense>
        <ForgotPasswordForm />
      </Suspense>
    </main>
  );
}
