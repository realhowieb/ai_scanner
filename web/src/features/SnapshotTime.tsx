"use client";

import { api, unwrap } from "@/api/client";
import { Freshness } from "@/components/ui";
import { useApi } from "@/hooks/useApi";

/** Use the server's schedule, never elapsed hours, to label a current snapshot. */
export function SnapshotTime({ at, archived = false }: { at?: string | null; archived?: boolean }) {
  const latest = useApi(at && !archived ? "snapshot-schedule" : null, (signal) =>
    unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: 1 } }, signal })));
  const same = at && latest.data?.scan_at && Date.parse(at) === Date.parse(latest.data.scan_at);
  const state = same ? latest.data?.freshness?.state : undefined;
  const stale = same ? state === "stale" ? true : state === "fresh" ? false : latest.data?.stale : undefined;
  return <Freshness at={at} label={archived ? "Archived snapshot" : "Scan"} stale={stale} />;
}
