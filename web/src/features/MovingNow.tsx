"use client";

// Day Trader "Moving now" (Pro): stocks HSF spotted today as an intraday move started
// (+3% vs yesterday's close on 2x the usual volume for the time of day, from 9:40 AM ET),
// with when and at what price they were first spotted. From /v1/day-trader/moving-now.
import { useEffect, useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, Empty, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime, num, pct, price } from "@/lib/format";

export const MOVING_NOW_REFRESH_MS = 2 * 60_000;   // the API checks about every 2 minutes
export const MOVING_NOW_SHOWN = 10;

function tone(v: number | null | undefined): string {
  return v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
}

/** `onCheck` puts the spotted tickers into the Day Trader table (typed-tickers source). */
export function MovingNow({ onCheck }: { onCheck?: (tickers: string[]) => void }) {
  const [tick, setTick] = useState(0);
  const { data, error } = useApi("dt-moving-now", (signal) =>
    unwrap(api.GET("/v1/day-trader/moving-now", { params: { query: { limit: 40 } }, signal })), tick);
  const live = !!data && data.state === "open";
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => { if (document.visibilityState === "visible") setTick((n) => n + 1); }, MOVING_NOW_REFRESH_MS);
    return () => clearInterval(t);
  }, [live]);
  const [all, setAll] = useState(false);

  if (error && !data) return null;   // the table below still works; this card is an extra
  const rows = data?.rows ?? [];
  const shown = all ? rows : rows.slice(0, MOVING_NOW_SHOWN);
  const today = data?.active || data?.state === "open";
  const title = today ? "Moving now" : "Spotted today";
  return (
    <Card title={title} id="moving-now" className="moving-now"
      aside={data ? `Checked ${etTime(data.as_of, false)}` : undefined}>
      {!data ? <p className="cap">Loading…</p>
        : rows.length === 0 ? (
          <Empty title={data.state === "premarket" || (data.state === "open" && !data.active)
            ? "Spotting starts at 9:40 AM ET."
            : data.active ? "Nothing spotted yet today." : "Nothing was spotted this session."}>
            Stocks show up here the first time in a session they trade 3% or more above yesterday&apos;s close on at least twice their usual volume for the time of day.
          </Empty>
        ) : (
          <>
            <p className="cap">{data.spotted_count} spotted today. Strongest volume first; times show when HSF first spotted each one.</p>
            <ul className="rows" aria-label="Stocks spotted as a move started">
              {shown.map((r) => (
                <li key={r.ticker} className="row mn-row">
                  <TickerLink ticker={r.ticker} named />
                  <span className="cap nowrap">spotted {etTime(r.spotted_at, false)} at <span className="mono">{price(r.spotted_price)}</span></span>
                  <span className="grow" />
                  <span className="mono nowrap">{price(r.last)}</span>
                  <span className={`mono strong move nowrap ${tone(r.since_spotted_pct)}`} title="Move since first spotted">{pct(r.since_spotted_pct)} since</span>
                  <span className="cap nowrap" title="Today's volume vs usual for this time of day">{num(r.volume_pace)}x vol</span>
                </li>
              ))}
            </ul>
            <div className="row-actions">
              {rows.length > MOVING_NOW_SHOWN && <button type="button" className="btn btn-sm" onClick={() => setAll((v) => !v)}>
                {all ? `Show top ${MOVING_NOW_SHOWN}` : `Show all ${rows.length}`}</button>}
              {onCheck && <button type="button" className="btn btn-sm" onClick={() => onCheck(shown.map((r) => r.ticker))}>Check these in the table</button>}
            </div>
          </>
        )}
      <p className="cap">
        Spotted the first time a liquid U.S. stock trades 3%+ above yesterday&apos;s close on 2x its usual volume for the time of day, checked about every 2 minutes from 9:40 AM ET.
        In a 16-session test this caught moves hours before the scans, though usually only a little cheaper, and showed no proven edge beyond the same day. Not a prediction.
      </p>
    </Card>
  );
}
