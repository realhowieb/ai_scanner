import type { Metadata } from "next";

import { PolicyPage } from "@/features/PolicyPage";
import { loadPolicy, versionParam } from "@/server/policy";

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "The terms for using HSF AI: accounts, plans and renewals, cancelling, acceptable use and disclaimers.",
  robots: { index: false }, // until the text is approved
};

export default async function Page({ searchParams }: Props) {
  return <PolicyPage loaded={await loadPolicy("terms", versionParam((await searchParams).version))} />;
}
