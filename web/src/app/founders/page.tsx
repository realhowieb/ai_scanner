import type { Metadata } from "next";

import { FoundersWallView } from "@/features/FoundersPublic";
import { loadFoundersWall } from "@/server/publicFounder";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Founders Wall",
  description: "The Founding 100 of HSFinest.AI: every founder seat by number. Names appear only when a founder chooses.",
};

export default async function Page() {
  return <FoundersWallView wall={await loadFoundersWall()} />;
}
