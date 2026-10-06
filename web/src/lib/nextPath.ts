/** Only same-site paths are allowed after sign-in (no open redirects). */
export function safeNext(raw: string | null | undefined): string {
  if (!raw || !raw.startsWith("/") || raw.startsWith("//") || raw.startsWith("/\\") || raw.startsWith("/api/")) return "/today";
  return raw;
}
