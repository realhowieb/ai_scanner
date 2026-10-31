import { publicAuth } from "@/server/auth";
import { upstreamFor } from "@/server/upstream";

export const dynamic = "force-dynamic";

export function POST(req: Request): Promise<Response> {
  return publicAuth(req, "email-link", upstreamFor(req));
}
