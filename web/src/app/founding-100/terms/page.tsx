import type { Metadata } from "next";

import { PolicyPage } from "@/features/PolicyPage";
import { loadPolicy, versionParam } from "@/server/policy";

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Founding Membership Terms",
  description: "The terms for HSF AI Founder Pro and Founder Elite: a one-time payment for five calendar years, with no automatic renewal.",
  robots: { index: false }, // until the text is approved and the program opens
};

export default async function Page({ searchParams }: Props) {
  return <PolicyPage loaded={await loadPolicy("founding_terms", versionParam((await searchParams).version))} />;
}
