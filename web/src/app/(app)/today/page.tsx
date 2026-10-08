"use client";

import { api, unwrap } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui";
import { PriceTape } from "@/features/PriceTape";
import { TodayView } from "@/features/TodayView";
import { useApi } from "@/hooks/useApi";

export default function TodayPage() {
  const { data, error, loading, reload } = useApi("today", (signal) => unwrap(api.GET("/v1/today", { signal })));
  return (
    <div className="stack">
      <PriceTape />
      {error && !data ? <ErrorState error={error} onRetry={reload} what="Today" />
        : loading && !data ? <Skeleton rows={8} label="Loading Today" />
        : data ? <TodayView data={data} /> : null}
    </div>
  );
}
