"use client";

import { api, unwrap } from "@/api/client";
import { Card, Empty, Locked, Pill, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useSession } from "@/session/SessionProvider";

import { SectionFailed } from "./TodayView";

const TIMES: Record<string, string> = { bmo: "Before open", amc: "After close" };
const ROW_LIMIT = 8;

function when(date: string | null | undefined, days: number | null | undefined): string {
  if (days === 0) return "Today";
  if (days === 1) return "Tomorrow";
  if (!date) return "Date unavailable";
  const d = new Date(`${date}T12:00:00Z`);
  return d.toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" });
}

/** Earnings in the next 7 days for the user's watchlist and today's top setups (Pro).
 * `tickers` is null until the page knows which names to ask about. */
export function EarningsCard({ tickers, watched }: { tickers: string[] | null; watched: Set<string> }) {
  const { can } = useSession();
  const allowed = can("can_earnings");
  const key = allowed && tickers && tickers.length > 0 ? `earnings:${tickers.join(",")}` : null;
  const { data, error } = useApi(key, (signal) =>
    unwrap(api.GET("/v1/earnings", { params: { query: { days: 7, tickers: tickers!.join(",") } }, signal })));

  if (!allowed) {
    return (
      <Card title="Earnings this week" id="earn">
        <Locked title="Earnings timing is part of Pro" plan="pro">See which names on your watchlist and in today&apos;s top setups report this week.</Locked>
      </Card>
    );
  }
  if (tickers === null) return null;
  if (error && !data) return <SectionFailed name="Earnings this week" />;
  const rows = data ?? [];
  return (
    <Card title="Earnings this week" id="earn" aside="Watchlist and top setups">
      {tickers.length > 0 && !data ? (
        <p className="cap">Loading…</p>
      ) : rows.length === 0 ? (
        <Empty title="No earnings this week for your watchlist or today's top setups." />
      ) : (
        <ul className="rows">
          {rows.slice(0, ROW_LIMIT).map((r) => (
            <li key={`${r.ticker}-${r.earnings_date}`} className="row">
              <TickerLink ticker={r.ticker} />
              <span className="strong grow">{when(r.earnings_date, r.days_until)}</span>
              <span className="cap nowrap">{(r.time && TIMES[r.time]) || "Time TBA"}</span>
              {watched.has(r.ticker) && <Pill>Watchlist</Pill>}
            </li>
          ))}
        </ul>
      )}
      {rows.length > ROW_LIMIT && <p className="cap">+{rows.length - ROW_LIMIT} more this week</p>}
    </Card>
  );
}
