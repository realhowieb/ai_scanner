// Liveness for the host's health check: answers without calling the API (an API
// outage must not make the host restart a healthy web server). `build` is the commit
// this server was deployed from, so the post-promote live check can wait for it.
export const dynamic = "force-dynamic";

export function GET(): Response {
  const build = (process.env.RENDER_GIT_COMMIT || "").trim().slice(0, 12) || null;
  return new Response(JSON.stringify({ ok: true, build }), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
