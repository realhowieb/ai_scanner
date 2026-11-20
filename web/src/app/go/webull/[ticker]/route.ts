import { upstreamFor } from "@/server/upstream";
import { webullTarget } from "@/server/webullLink";

export const dynamic = "force-dynamic";

/** /go/webull/AAPL -> the stock's Webull quote page (or Webull's markets page if unknown). */
export async function GET(req: Request, ctx: { params: Promise<{ ticker: string }> }): Promise<Response> {
  const to = await webullTarget((await ctx.params).ticker, upstreamFor(req));
  return new Response(null, { status: 302, headers: { location: to, "cache-control": "private, max-age=3600", "referrer-policy": "no-referrer" } });
}
