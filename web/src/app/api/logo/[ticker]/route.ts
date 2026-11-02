// Company logo for the Earnings calendar, fetched server-side from FMP's public image
// URL so the browser stays same-origin (the CSP allows only 'self' images) and the
// viewer's IP never reaches a third party. A missing logo answers 404 and the tile
// shows the ticker's first letter instead.
export const dynamic = "force-dynamic";

const TICKER = /^[A-Z]{1,6}(-[A-Z]{1,2})?$/;
const WEEK = 7 * 24 * 3600;
// Raster only: an SVG served from our origin could carry script if opened directly.
const TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

export async function GET(_req: Request, ctx: { params: Promise<{ ticker: string }> }): Promise<Response> {
  const ticker = decodeURIComponent((await ctx.params).ticker || "").toUpperCase().replace(".", "-");
  const miss = () => new Response(null, { status: 404, headers: { "cache-control": "public, max-age=86400" } });
  if (!TICKER.test(ticker)) return miss();
  try {
    const r = await fetch(`https://financialmodelingprep.com/image-stock/${ticker}.png`, {
      signal: AbortSignal.timeout(4000), next: { revalidate: WEEK },
    });
    const type = (r.headers.get("content-type") ?? "").split(";")[0]?.trim().toLowerCase() ?? "";
    if (!r.ok || !TYPES.has(type)) return miss();
    return new Response(await r.arrayBuffer(), {
      headers: { "content-type": type, "cache-control": `public, max-age=${WEEK}, immutable`, "x-content-type-options": "nosniff" },
    });
  } catch {
    return miss();
  }
}
