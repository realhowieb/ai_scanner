"use client";

import { api, unwrap } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui";
import { TodayView } from "@/features/TodayView";
import { useApi } from "@/hooks/useApi";

export default function TodayPage() {
  const { data, error, loading, reload } = useApi("today", (signal) => unwrap(api.GET("/v1/today", { signal })));
  if (error && !data) return <ErrorState error={error} onRetry={reload} what="Today" />;
  if (loading && !data) return <Skeleton rows={8} label="Loading Today" />;
  return data ? <TodayView data={data} /> : null;
}
