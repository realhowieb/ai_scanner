import { publicApi } from "@/server/public";
import { upstream } from "@/server/upstream";

export const dynamic = "force-dynamic";

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: Request, ctx: Ctx): Promise<Response> {
  return publicApi(req, (await ctx.params).path, upstream);
}

export async function POST(req: Request, ctx: Ctx): Promise<Response> {
  return publicApi(req, (await ctx.params).path, upstream);
}
