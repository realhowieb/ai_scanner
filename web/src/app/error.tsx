"use client";

import { PageError } from "@/components/PageError";

export default function RootError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  return (
    <main className="center">
      <PageError error={error} retry={retry} home="/" />
    </main>
  );
}
