import type { Metadata } from "next";

import { PolicyPage } from "@/features/PolicyPage";
import { loadPolicy, versionParam } from "@/server/policy";

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Founding Membership Refund Policy",
  description: "The 14-day refund guarantee for HSF AI founding memberships, upgrade refunds and the five-year service continuity refund.",
  robots: { index: false }, // until the text is approved and the program opens
};

export default async function Page({ searchParams }: Props) {
  return <PolicyPage loaded={await loadPolicy("founding_refunds", versionParam((await searchParams).version))} />;
}
