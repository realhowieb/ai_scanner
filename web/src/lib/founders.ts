const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** "October 2026" from an ISO date or "2026-10"; null when it isn't one. */
export function monthYear(iso: string | null | undefined): string | null {
  const m = /^(\d{4})-(\d{2})/.exec(iso || "");
  const i = m ? Number(m[2]) - 1 : -1;
  return m && i >= 0 && i < 12 ? `${MONTHS[i]} ${m[1]}` : null;
}
