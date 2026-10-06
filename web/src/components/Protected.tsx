"use client";

import type { ReactNode } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, Skeleton } from "@/components/ui";
import { SessionProvider, useSession } from "@/session/SessionProvider";

function Gate({ children }: { children: ReactNode }) {
  const { me, loading, error, reload } = useSession();
  if (loading && !me) return <Skeleton rows={6} label="Restoring your session" />;
  if (error && !me) return <ErrorState error={error} onRetry={reload} what="your account" />;
  return <>{children}</>;
}

export function Protected({ children }: { children: ReactNode }) {
  return (
    <SessionProvider>
      <AppShell>
        <Gate>{children}</Gate>
      </AppShell>
    </SessionProvider>
  );
}
