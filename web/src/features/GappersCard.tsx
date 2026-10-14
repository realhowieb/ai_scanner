"use client";

import { useEffect } from "react";

import { api, unwrap } from "@/api/client";
import { Card, Empty, Freshness, Locked, TickerLink } from "@/components/ui";
import { compact, pct, price } from "@/lib/format";
import { useApi } from "@/hooks/useApi";
import { useSession } from "@/session/SessionProvider";

import { SectionFailed } from "./TodayView";

export const GAPPERS_REFRESH_MS = 3 * 60_000;   // the API rebuilds the list every 3 minutes
const ROW_LIMIT = 10;

/** Today's pre-market gappers across S&P 500 + NASDAQ (Pro), shown before the open. */
export function GappersCard() {
  const { can } = useSession();
  const allowed = can("can_premarket");
  const { data, error, reload } = useApi(allowed ? "pm-gappers" : null, (signal) =>
    unwrap(api.GET("/v1/premarket/gappers", { params: { query: { limit: 25 } }, signal })));
  useEffect(() => {
    if (!allowed) return;
    const t = setInterval(() => { if (document.visibilityState === "visible") reload(); }, GAPPERS_REFRESH_MS);
    return () => clearInterval(t);
  }, [allowed, reload]);

  if (!allowed) {
    return (
      <Card title="Pre-market gappers" id="gap">
        <Locked title="Pre-market gappers are part of Pro" plan="pro">See which stocks are gapping before the open across the S&amp;P 500 and NASDAQ, with pre-market volume.</Locked>
      </Card>
    );
  }
  if (error && !data) return <SectionFailed name="Pre-market gappers" />;
  if (data && !data.active) return null;
  const rows = data?.rows ?? [];
  const partial = data?.feed === "iex";
  return (
    <Card title="Pre-market gappers" id="gap" aside={data ? <Freshness at={data.as_of} label="Updated" staleCheck={false} /> : undefined}>
      {!data ? (
        <p className="cap">Loading…</p>
      ) : rows.length === 0 ? (
        <Empty title="No pre-market gappers yet today.">Names show up here once they trade before the open and move 2% or more.</Empty>
      ) : (
        <ul className="rows">
          {rows.slice(0, ROW_LIMIT).map((g) => (
            <li key={g.ticker} className="row">
              <TickerLink ticker={g.ticker} />
              <span className={`mono strong move ${g.pct > 0 ? "up" : g.pct < 0 ? "down" : ""}`}>{pct(g.pct)}</span>
              <span className="mono cap grow">{price(g.last)}</span>
              {g.pm_volume != null && <span className="cap nowrap">{compact(g.pm_volume)} sh</span>}
            </li>
          ))}
        </ul>
      )}
      <p className="cap">
        Latest pre-market trade vs yesterday&apos;s close, S&amp;P 500 and NASDAQ, refreshed every few minutes.
        {partial ? " Volume is from the IEX feed, a slice of all pre-market trading." : ""}
      </p>
    </Card>
  );
}
