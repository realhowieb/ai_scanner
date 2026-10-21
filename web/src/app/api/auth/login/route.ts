import { login } from "@/server/auth";
import { upstreamFor } from "@/server/upstream";

export const dynamic = "force-dynamic";

export function POST(req: Request): Promise<Response> {
  return login(req, upstreamFor(req));
}
