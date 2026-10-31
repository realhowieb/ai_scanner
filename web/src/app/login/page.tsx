import type { Metadata } from "next";
import { Suspense } from "react";

import { LoginAside } from "./LoginAside";
import { LoginForm } from "./LoginForm";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <main className="auth-split">
      <LoginAside />
      <div className="center auth-main">
        <Suspense>
          <LoginForm />
        </Suspense>
      </div>
    </main>
  );
}
