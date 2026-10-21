/** Only the selected ingestion origin is added to the browser's connection policy. */
export function posthogHost(value: string | undefined): string {
  return value === "https://eu.i.posthog.com" ? value : "https://us.i.posthog.com";
}
