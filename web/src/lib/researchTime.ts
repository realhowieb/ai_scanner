/** State outcomes use elapsed wall-clock offsets, not trading-day returns. */
export function stateHorizon(horizon?: string): string {
  const labels: Record<string, string> = {
    NEXT: "the next comparable HSF observation",
    H24: "the first comparable observation at least 24 elapsed hours later",
    H72: "the first comparable observation at least 72 elapsed hours later",
    H120: "the first comparable observation at least 120 elapsed hours later",
  };
  return horizon ? labels[horizon] ?? `the recorded ${horizon} horizon (definition unavailable)` : "an unspecified horizon";
}

/** Only explain a weekend gap when the server confirms the schedule is current. */
export function scheduledSession(at?: string | null, checked?: string | null, onSchedule = false): string | null {
  if (!at || !checked || !onSchedule) return null;
  const scan = new Date(at), now = new Date(checked);
  if (!Number.isFinite(scan.getTime()) || !Number.isFinite(now.getTime()) || scan > now) return null;
  const day = now.toLocaleDateString("en-US", { timeZone: "America/New_York", weekday: "long" });
  if (day !== "Saturday" && day !== "Sunday") return null;
  const label = scan.toLocaleDateString("en-US", { timeZone: "America/New_York", weekday: "long", month: "short", day: "numeric" });
  return `Latest scheduled scan: ${label} · market closed for the weekend`;
}
