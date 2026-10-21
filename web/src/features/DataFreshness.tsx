import type { Schemas } from "@/api/client";
import { scheduledSession } from "@/lib/researchTime";
import { etTime } from "@/lib/format";

const labels = { fresh: "On schedule", stale: "Scan delayed", partial: "Timestamp coverage incomplete", unavailable: "Snapshot unavailable" };

/** Server calendar rules decide freshness; a time the server doesn't know is left out, not guessed. */
export function DataFreshness({ info }: { info?: Schemas["DataFreshness"] | null }) {
  if (!info) return null; // Compatible with older API versions.
  const sessionNote = scheduledSession(info.last_successful_scan_at, info.checked_at, info.state === "fresh");
  const time = (value?: string | null) => value ? etTime(value) : "Unavailable";
  return (
    <details aria-label="Data freshness" className="data-fresh">
      <summary><strong><span className="wide-text">Data freshness: </span>{labels[info.state]}</strong> · <span className="wide-text">Saved scan </span>{time(info.last_successful_scan_at)}</summary>
      {sessionNote && <p className="cap">{sessionNote}</p>}
      <p className="cap">Schedule status and price age are separate. Saved scan prices are not live quotes.</p>
      <div>Last successful scan (saved): {time(info.last_successful_scan_at)}</div>
      {info.scan_completed_at && <div>Scan completed: {time(info.scan_completed_at)}</div>}
      <div>Market data as of: {time(info.market_data_at)}</div>
      {info.market_data_note && <p className="cap">{info.market_data_note}</p>}
    </details>
  );
}
