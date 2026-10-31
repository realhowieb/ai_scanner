import type { Metadata } from "next";
import { Suspense } from "react";

import { SignedOut } from "./SignedOut";

export const metadata: Metadata = { title: "Signed out", robots: { index: false } };

export default function SignedOutPage() {
  return (
    <main className="center">
      <Suspense>
        <SignedOut />
      </Suspense>
    </main>
  );
}
