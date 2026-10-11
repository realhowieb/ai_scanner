"use client";

import { PageError } from "@/components/PageError";

// Inside the signed-in layout, so the navigation stays usable when one page crashes.
export default function AppError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  return <PageError error={error} retry={retry} />;
}
