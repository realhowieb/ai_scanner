import { after } from "next/server";

import { proxy } from "@/server/bff";
import { flushPostHogLogs, logProxyRequestStarted, logProxyResponse, posthogLogsEnabled } from "@/server/posthogLogs";
import { upstreamFor } from "@/server/upstream";

export const dynamic = "force-dynamic";

type Ctx = { params: Promise<{ path: string[] }> };

async function handle(req: Request, ctx: Ctx): Promise<Response> {
  const { path } = await ctx.params;
  logProxyRequestStarted(req.method);
  const response = await proxy(req, path.join("/"), upstreamFor(req));
  logProxyResponse(req.method, response.status);
  if (posthogLogsEnabled()) {
    try { after(flushPostHogLogs); } catch { /* Logging must not block a response. */ }
  }
  return response;
}

export { handle as GET, handle as POST, handle as PATCH, handle as DELETE };
