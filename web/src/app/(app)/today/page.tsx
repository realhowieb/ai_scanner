"use client";

import { api, unwrap } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui";
import { PriceTape } from "@/features/PriceTape";
import { TodayView } from "@/features/TodayView";
import { useApi } from "@/hooks/useApi";
import { readMarker, writeMarker } from "@/lib/lastVisit";

export default function TodayPage() {
  const { data, error, loading, reload } = useApi("today", (signal) => unwrap(api.GET("/v1/today", { signal })));
  const mine = useApi("today-me", async (signal) => {
    const out = await unwrap(api.GET("/v1/today/me", { params: { query: readMarker() }, signal }));
    writeMarker(out.new_since?.marker);
    return out;
  });
  return (
    <div className="stack">
      <PriceTape />
      {error && !data ? <ErrorState error={error} onRetry={reload} what="Today" />
        : loading && !data ? <Skeleton rows={8} label="Loading Today" />
        : data ? <TodayView data={data} mine={mine.data} mineFailed={!!mine.error && !mine.data} /> : null}
    </div>
  );
}
