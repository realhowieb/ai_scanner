"use client";

import { useParams } from "next/navigation";

import { api, unwrap } from "@/api/client";
import { TICKER_RE } from "@/components/AppShell";
import { Card, Empty, ErrorState, Skeleton } from "@/components/ui";
import { StockView } from "@/features/StockView";
import { useApi } from "@/hooks/useApi";
import { useSession } from "@/session/SessionProvider";

export default function StockPage() {
  const params = useParams<{ ticker: string }>();
  const ticker = decodeURIComponent(params.ticker || "").toUpperCase();
  const valid = TICKER_RE.test(ticker);
  const { can } = useSession();
  const { data, error, loading, reload } = useApi(valid ? `stock:${ticker}` : null, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}", { params: { path: { ticker } }, signal })));
  if (!valid) return <Card><Empty title="That isn't a ticker symbol.">Search for a symbol like AAPL.</Empty></Card>;
  if (error && !data) return <ErrorState error={error} onRetry={reload} what={ticker} />;
  // Keep the page (and any open dialog) while it refreshes after a save; blank only for a new ticker.
  if (!data || (loading && data.ticker !== ticker)) return <Skeleton rows={8} label={`Loading ${ticker}`} />;
  return <StockView s={data} premium={can("can_early_breakout")} onChanged={reload} />;
}
