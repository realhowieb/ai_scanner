import { proxy } from "@/server/bff";
import { upstream } from "@/server/upstream";

export const dynamic = "force-dynamic";

type Ctx = { params: Promise<{ path: string[] }> };

async function handle(req: Request, ctx: Ctx): Promise<Response> {
  const { path } = await ctx.params;
  return proxy(req, path.join("/"), upstream);
}

export { handle as GET, handle as POST, handle as PATCH, handle as DELETE };
