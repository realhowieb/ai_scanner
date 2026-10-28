import type { Metadata } from "next";

import { PolicyPage } from "@/features/PolicyPage";
import { loadPolicy, versionParam } from "@/server/policy";

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "What personal information HSF AI collects, why, who it is shared with and the choices you have.",
  robots: { index: false }, // until the text is approved
};

export default async function Page({ searchParams }: Props) {
  return <PolicyPage loaded={await loadPolicy("privacy", versionParam((await searchParams).version))} />;
}
