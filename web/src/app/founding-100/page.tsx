import type { Metadata } from "next";

import { FoundingPageView } from "@/features/FoundingPage";
import { publicMetadata } from "@/lib/seo";
import { loadPublicFounding } from "@/server/publicFounder";

export const dynamic = "force-dynamic";

const TITLE = "The Founding 100: one payment, five years of HSF";
const DESCRIPTION = "100 numbered founding memberships of HSFinest.com. Founder Pro $499 or Founder Elite $999, paid once for five years. No renewals, 14-day refund guarantee.";

export const metadata: Metadata = publicMetadata("/founding-100", TITLE, DESCRIPTION, "/founding-100/opengraph-image");

export default async function Page() {
  return <FoundingPageView initial={await loadPublicFounding()} />;
}
