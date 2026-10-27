"use client";

import type { ApiError, Schemas } from "@/api/client";
import { RefreshStatus } from "@/components/ui";
import { scheduledSession } from "@/lib/researchTime";
import { etTime } from "@/lib/format";

function ageLabel(at: string | null | undefined, checked: string | null | undefined): string | null {
  if (!at || !checked) return null;
  const minutes = Math.floor((Date.parse(checked) - Date.parse(at)) / 60_000);
  if (!Number.isFinite(minutes) || minutes < 0) return null;
  if (minutes < 1) return "less than a minute old";
  if (minutes < 60) return `${minutes}m old`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h${minutes % 60 ? ` ${minutes % 60}m` : ""} old`;
  return `${Math.floor(minutes / 1440)}d old`;
}

/** A scheduled snapshot can be on time while its prices are hours old. */
export function ScannerDataStatus({ info, scanAt, stale, checkedAt, refreshing, error, reload }: {
  info?: Schemas["DataFreshness"] | null;
  scanAt?: string | null;
  stale?: boolean | null;
  checkedAt: number | null;
  refreshing: boolean;
  error: ApiError | null;
  reload: () => void;
}) {
  const delayed = info?.state === "stale" || stale === true;
  const label = delayed ? "Scan delayed" : info?.state === "fresh" ? "On schedule"
    : info?.state === "partial" ? "Timestamp coverage incomplete"
    : info?.state === "unavailable" ? "Snapshot unavailable" : "Saved snapshot";
  const asOf = info?.market_data_at;
  const savedAt = info?.last_successful_scan_at ?? scanAt;
  const checked = checkedAt ? new Date(checkedAt).toISOString() : info?.checked_at;
  const sessionNote = scheduledSession(savedAt, checked, info?.state === "fresh" && !delayed);
  // The browser's own check time: an unchanged answer (304) keeps the server's older checked_at.
  const age = ageLabel(asOf ?? savedAt, checked);
  const time = (value?: string | null) => value ? etTime(value) : "Unavailable";
  return <section className={`scanner-data-status${delayed ? " is-delayed" : ""}`} aria-label="Scanner data status">
    <details>
      <summary>
        <span className="scanner-data-badge">{label}</span>
        <span className="scanner-data-time">{asOf ? `Snapshot prices as of ${time(asOf)}` : savedAt ? `Saved scan ${time(savedAt)} · Price time unavailable` : "Snapshot time unavailable"}{age && ` · ${age}`}
          {sessionNote && <span className="cap scanner-session-note">{sessionNote}</span>}
        </span>
        <span className="scanner-data-details">Details</span>
      </summary>
      <div className="scanner-data-body">
        <p>These are saved scan prices. Refresh checks for a newer scan; it does not fetch live prices.</p>
        <dl>
          <div><dt>Scan saved</dt><dd>{time(savedAt)}</dd></div>
          <div><dt>Scan completed</dt><dd>{time(info?.scan_completed_at)}</dd></div>
          <div><dt>Market data as of</dt><dd>{time(asOf)}</dd></div>
          <div><dt>Last checked for a newer scan</dt><dd>{time(checked)}</dd></div>
        </dl>
        {info?.state === "fresh" && !delayed && <p>On schedule means no expected scan has been missed. It does not mean the prices are live.</p>}
        {delayed && <p>A scheduled scan is overdue. Scores and prices still come from the saved snapshot.</p>}
        {info?.market_data_note && <p>{info.market_data_note}</p>}
      </div>
    </details>
    {delayed && <p className="scanner-data-warning" role="status">A scheduled scan is overdue. Showing the last saved snapshot.</p>}
    {(refreshing || error) && <RefreshStatus updatedAt={null} refreshing={refreshing} error={error} onRetry={reload} />}
  </section>;
}
