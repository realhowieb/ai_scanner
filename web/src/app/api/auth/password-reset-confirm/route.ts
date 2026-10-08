import { publicAuth } from "@/server/auth";
import { upstream } from "@/server/upstream";

export const dynamic = "force-dynamic";

export function POST(req: Request): Promise<Response> {
  return publicAuth(req, "password-reset-confirm", upstream);
}
