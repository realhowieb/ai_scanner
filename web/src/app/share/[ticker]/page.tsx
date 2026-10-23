import type { Metadata } from "next";

import { SharedStock } from "@/features/SharedStock";
import { pct, price, setupLabel } from "@/lib/format";
import { loadPublicStock } from "@/server/publicStock";

type Props = { params: Promise<{ ticker: string }> };

// Signed-in visitors never see this: the proxy sends /share/X to /stocks/X for them.
export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const s = await loadPublicStock((await params).ticker);
  if (!s) return { title: "Stock not found" };
  const move = s.price != null ? ` ${price(s.price)} (${pct(s.change_pct)})` : "";
  const title = s.ranked && s.hsf_score != null
    ? `${s.ticker}: HSF Score ${s.hsf_score}, ${setupLabel(s.setup)} setup`
    : `${s.ticker}${s.name ? ` · ${s.name}` : ""}`;
  const description = s.ranked
    ? `${s.ticker}${move} is in HSF's latest ranked setups. ${s.reason ?? ""}`.trim()
    : `${s.ticker}${move}. See how HSF scores it and get alerted when it becomes a ranked setup.`;
  return { title: { absolute: `${title} · HSFinest.AI` }, description, openGraph: { title, description, type: "website", siteName: "HSFinest.AI" } };
}

export default async function SharePage({ params }: Props) {
  const raw = (await params).ticker;
  return <SharedStock stock={await loadPublicStock(raw)} asked={decodeURIComponent(raw).toUpperCase().slice(0, 10)} />;
}
