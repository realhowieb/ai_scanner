/** Seconds to wait from a Retry-After header (delta-seconds or HTTP date), or null. */
export function parseRetryAfter(value: string | null | undefined, nowMs = Date.now()): number | null {
  if (!value) return null;
  const v = value.trim();
  if (/^\d+$/.test(v)) return Math.min(3600, parseInt(v, 10));
  const when = Date.parse(v);
  if (Number.isNaN(when)) return null;
  return Math.max(0, Math.min(3600, Math.ceil((when - nowMs) / 1000)));
}
