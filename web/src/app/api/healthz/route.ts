// Liveness for the host's health check: answers without calling the API (an API
// outage must not make the host restart a healthy web server).
export const dynamic = "force-dynamic";

export function GET(): Response {
  return new Response(JSON.stringify({ ok: true }), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
