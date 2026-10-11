import type { Schemas } from "@/api/client";
import { etTime } from "@/lib/format";

const labels = { fresh: "Fresh", stale: "Stale", partial: "Partially available", unavailable: "Unavailable" };

/** Server calendar rules decide freshness; a time the server doesn't know is left out, not guessed. */
export function DataFreshness({ info }: { info?: Schemas["DataFreshness"] | null }) {
  if (!info) return null; // Compatible with older API versions.
  const time = (value?: string | null) => value ? etTime(value) : "Unavailable";
  return (
    <div aria-label="Data freshness" className="data-fresh">
      <strong>Data freshness: {labels[info.state]}</strong>
      <div>Last successful scan (saved): {time(info.last_successful_scan_at)}</div>
      {info.scan_completed_at && <div>Scan completed: {time(info.scan_completed_at)}</div>}
      {info.market_data_at && <div>Market data as of: {time(info.market_data_at)}</div>}
    </div>
  );
}
