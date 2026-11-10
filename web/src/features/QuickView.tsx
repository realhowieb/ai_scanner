"use client";

// Scanner quick view: a side panel with a stock's chart, why it ranks and its risks, so a
// row can be checked without leaving the Scanner. Reads the same GET /v1/stocks/{ticker}
// as the stock page (same cache key), so opening the full page afterwards is instant.
import Link from "@/components/Link";
import { Dialog as Radix } from "radix-ui";
import { useRef } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { TickerLogo } from "@/components/TickerLogo";
import { Empty, ErrorState, Pill, ScoreBadge, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useCompanyName } from "@/lib/companyNames";
import { pct, price, setupLabel } from "@/lib/format";

import { PriceChart } from "./PriceChart";
import { SaveToWatchlistButton } from "./SaveToWatchlist";
import { RankMove, sinceScan } from "./SetupTable";
import { earningsDate, movementText, signalEvents } from "./StockView";

type Row = Schemas["ScanSetup"] & { scan_last?: number | null };
type Stock = Schemas["StockDetail"];

const upDown = (v: number | null | undefined) => (v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "";

function Body({ row, s, pro, compared }: { row: Row; s: Stock; pro: boolean; compared?: boolean }) {
  const move = sinceScan(row);
  const href = `/stocks/${encodeURIComponent(row.ticker)}`;
  return <div className="stack-sm quick-view-body">
    <div className="quote">
      <span className="mono big">{price(row.last)}</span>
      <span className={`mono ${upDown(row.chg_pct)}`}>{pct(row.chg_pct)}</span>
      {move != null && <span className={`cap mono ${upDown(move)}`}>{pct(move, 1)} since scan</span>}
    </div>
    <div className="row-actions">
      <ScoreBadge score={row.score} />
      {row.primary_setup && <Pill>{setupLabel(row.primary_setup)}</Pill>}
      {row.rank != null && <span className="cap mono">#{row.rank} in the scan</span>}
      <RankMove row={row} compared={compared} />
      {row.sector && <span className="cap">{row.sector}</span>}
    </div>
    {s.bars.length >= 2
      ? <PriceChart bars={s.bars} events={signalEvents(s)} earningsDate={earningsDate(s.earnings_days)} volumeFeed={s.volume_feed} />
      : <Empty title="No price history cached for this ticker." />}
    <p className="cap">{movementText(s)}</p>
    <section aria-label="Why it ranks">
      <h3 className="h3">Why it ranks</h3>
      {s.reasons.length ? <ul className="bullets">{s.reasons.slice(0, 4).map((r) => <li key={r}>{r}</li>)}</ul> : <p className="cap">No reason recorded.</p>}
    </section>
    {s.risks.length > 0 && <section aria-label="Risks">
      <h3 className="h3">Risks</h3>
      <ul className="bullets">{s.risks.slice(0, 3).map((r) => <li key={r}>{r}</li>)}</ul>
    </section>}
    {s.earnings_days != null && s.earnings_days >= 0 && s.earnings_days <= 14 && <p className="cap">
      Earnings {s.earnings_days === 0 ? "today" : s.earnings_days === 1 ? "tomorrow" : `in ${s.earnings_days} days`}.</p>}
    <div className="row-actions">
      <SaveToWatchlistButton ticker={row.ticker} />
      <Link className="btn btn-sm" href={`${href}#plan`}>{pro ? "Trade plan" : "Trade plan (Pro)"}</Link>
      <Link className="btn btn-sm btn-primary" href={href}>Open full page</Link>
    </div>
  </div>;
}

/** Opens when `row` is set; `onClose` clears it. */
export function QuickView({ row, onClose, pro, compared }: { row: Row | null; onClose: () => void; pro: boolean; compared?: boolean }) {
  const ticker = row?.ticker.toUpperCase() ?? "";
  const stock = useApi(row ? `stock:${ticker}` : null, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}", { params: { path: { ticker } }, signal })));
  const name = useCompanyName(ticker);
  const opener = useRef<HTMLElement | null>(null);
  const s = stock.data && stock.data.ticker === ticker ? stock.data : null;
  return (
    <Radix.Root open={row !== null} onOpenChange={(next) => { if (!next) onClose(); }}>
      <Radix.Portal>
        <Radix.Overlay className="backdrop drawer-backdrop">
          <Radix.Content className="drawer" aria-describedby={undefined}
            onOpenAutoFocus={() => { opener.current = document.activeElement as HTMLElement | null; }}
            onCloseAutoFocus={(e) => { const back = opener.current; if (back?.isConnected) { e.preventDefault(); back.focus(); } }}>
            <div className="dialog-head">
              <div className="title-row">
                {row && <TickerLogo ticker={ticker} size={28} />}
                <Radix.Title className="h2 mono">{ticker}</Radix.Title>
                {name && <span className="cap">{name}</span>}
              </div>
              <Radix.Close className="icon-btn" aria-label="Close">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
              </Radix.Close>
            </div>
            <div className="dialog-body">
              {row && (s ? <Body row={row} s={s} pro={pro} compared={compared} />
                : stock.error ? <ErrorState error={stock.error} onRetry={stock.reload} what={ticker} />
                : <Skeleton rows={6} label={`Loading ${ticker}`} />)}
            </div>
          </Radix.Content>
        </Radix.Overlay>
      </Radix.Portal>
    </Radix.Root>
  );
}
