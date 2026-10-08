import type { Metadata } from "next";

import { TrackRecordView } from "@/features/TrackRecordView";

export const metadata: Metadata = { title: "Track record" };

export default function TrackRecordPage() {
  return <TrackRecordView />;
}
