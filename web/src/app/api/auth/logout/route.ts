import { logout } from "@/server/auth";
import { upstream } from "@/server/upstream";

export const dynamic = "force-dynamic";

export function POST(req: Request): Promise<Response> {
  return logout(req, upstream);
}
