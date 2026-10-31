import type { Metadata } from "next";

import { FounderCardView } from "@/features/FoundersPublic";
import { loadFounderCard } from "@/server/publicFounder";

type Props = { params: Promise<{ code: string }> };

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const c = await loadFounderCard((await params).code);
  if (!c) return { title: "Founder card not found", robots: { index: false } };
  const title = `${c.badge ?? c.label} · HSFinest.AI`;
  const description = `One of 100 founding members of HSFinest.AI. ${c.label}.`;
  return { title: { absolute: title }, description, robots: { index: false },
    openGraph: { title, description, type: "website", siteName: "HSFinest.AI" } };
}

export default async function Page({ params }: Props) {
  const code = (await params).code;
  return <FounderCardView card={await loadFounderCard(code)} />;
}
