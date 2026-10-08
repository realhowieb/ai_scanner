"use client";

import { api, unwrap } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui";
import { EarningsCard } from "@/features/EarningsCard";
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
  // Earnings asks about the watchlist and today's top setups once both are known.
  const wl = mine.data?.watchlist;
  const watched = new Set([...(wl?.in_scan.map((r) => r.ticker) ?? []), ...(wl?.missing ?? [])]);
  const top = [...(data?.top_setups?.setups ?? []), ...(data?.top_setups?.also_ranked ?? [])].map((s) => s.ticker);
  const settled = !!data && (!!mine.data || !!mine.error);
  const tickers = settled ? [...new Set([...watched, ...top])].sort() : null;
  return (
    <div className="stack">
      <PriceTape />
      {error && !data ? <ErrorState error={error} onRetry={reload} what="Today" />
        : loading && !data ? <Skeleton rows={8} label="Loading Today" />
        : data ? <TodayView data={data} mine={mine.data} mineFailed={!!mine.error && !mine.data}
            side={<EarningsCard tickers={tickers} watched={watched} />} /> : null}
    </div>
  );
}
