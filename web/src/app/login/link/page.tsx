import type { Metadata } from "next";
import { Suspense } from "react";

import { EmailLinkSignIn } from "./EmailLinkSignIn";

export const metadata: Metadata = { title: "Sign in with your email link", robots: { index: false } };

export default function EmailLinkPage() {
  return (
    <main className="center">
      <Suspense>
        <EmailLinkSignIn />
      </Suspense>
    </main>
  );
}
