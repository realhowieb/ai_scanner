// "Oct 27 – Oct 31, 2026" for a recap week. Server-safe (no "use client" anywhere in
// the chain) so the /weekly server pages can call it — a client-module export
// cannot be invoked from a Server Component.
const SHORT = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const YEAR = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const d = (iso: string) => new Date(`${iso}T12:00:00Z`);

export function weekRange(start: string, end: string): string {
  return `${SHORT.format(d(start))} – ${YEAR.format(d(end))}`;
}
