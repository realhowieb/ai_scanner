import type { Metadata } from "next";

import { PublicTrackRecordView } from "@/features/PublicTrackRecord";
import { publicMetadata } from "@/lib/seo";
import { loadPublicRecord } from "@/server/publicRecord";

export const dynamic = "force-dynamic";

export const metadata: Metadata = publicMetadata("/results", "HSF track record: every top pick vs the S&P 500",
  "Every day's top 10 HSF setups, saved when the scan ran, and how each did over the next 5 trading days against the S&P 500. Losers included, updated daily.");

export default async function Page() {
  return <PublicTrackRecordView record={await loadPublicRecord()} />;
}
