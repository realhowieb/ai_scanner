/** Only same-site paths are allowed after sign-in (no open redirects). */
export function safeNext(raw: string | null | undefined): string {
  if (!raw || !raw.startsWith("/") || raw.startsWith("//") || raw.startsWith("/\\") || raw.startsWith("/api/")) return "/today";
  return raw;
}

/** `path` carrying the visitor's destination along (sign-in <-> sign-up), when it's a safe one. */
export function withNext(path: string, raw: string | null | undefined): string {
  const next = safeNext(raw);
  return next === "/today" ? path : `${path}?next=${encodeURIComponent(next)}`;
}
