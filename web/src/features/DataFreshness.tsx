import type { Schemas } from "@/api/client";
import { etTime } from "@/lib/format";

const labels = { fresh: "On schedule", stale: "Scan delayed", partial: "Timestamp coverage incomplete", unavailable: "Snapshot unavailable" };

/** Server calendar rules decide freshness; a time the server doesn't know is left out, not guessed. */
export function DataFreshness({ info }: { info?: Schemas["DataFreshness"] | null }) {
  if (!info) return null; // Compatible with older API versions.
  const time = (value?: string | null) => value ? etTime(value) : "Unavailable";
  return (
    <details aria-label="Data freshness" className="data-fresh">
      <summary><strong>Data freshness: {labels[info.state]}</strong> · Saved scan {time(info.last_successful_scan_at)}</summary>
      <p className="cap">Schedule status and price age are separate. Saved scan prices are not live quotes.</p>
      <div>Last successful scan (saved): {time(info.last_successful_scan_at)}</div>
      {info.scan_completed_at && <div>Scan completed: {time(info.scan_completed_at)}</div>}
      <div>Market data as of: {time(info.market_data_at)}</div>
      {info.market_data_note && <p className="cap">{info.market_data_note}</p>}
    </details>
  );
}
