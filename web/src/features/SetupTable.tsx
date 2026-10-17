"use client";

import Link from "next/link";
import { ResearchGlossary } from "@/components/ResearchGlossary";
import { useState } from "react";
import { useWorkspace } from "@/hooks/useWorkspace";
import { DEFAULT_COLUMNS, TablePreferences } from "./TablePreferences";

import type { Schemas } from "@/api/client";
import { Pill, ScoreBadge, ScoreBar, TickerLink } from "@/components/ui";
import { num, pct, price, probWithRank, setupLabel } from "@/lib/format";

import { SaveToWatchlistButton } from "./SaveToWatchlist";
import { DailySpark, RvolMeter, useDailySparks } from "./Sparks";

type Row = Schemas["ScanSetup"];

/** Ranked rows, as a table on wide screens and as cards on phones. Rows come ranked from the API. */
/** `numbered` = rows are in HSF rank order, so the position is the rank; off for other sort orders. */
/** The PreBreakout column shows only on Premium; other plans get one note under the rows instead of "Premium" per row. */
/** `sparks` adds each ticker's 20-session sparkline: for current scans only, not saved history. */
export function SetupTable({ rows, offset = 0, premium, numbered = true, sparks = false }: { rows: Row[]; offset?: number; premium: boolean; numbered?: boolean; sparks?: boolean }) {
  const prefs = useWorkspace();
  const [selected, setSelected] = useState<string[]>([]);
  const table = prefs.data?.table;
  const visible = (col: string) => !sparks || (table?.columns ?? DEFAULT_COLUMNS).some((c) => c === col);
  const toggle = (ticker: string) => setSelected((prev) => prev.includes(ticker) ? prev.filter((t) => t !== ticker) : prev.length < 4 ? [...prev, ticker] : prev);
  const choose = (ticker: string) => <input type="checkbox" aria-label={`Compare ${ticker}`} checked={selected.includes(ticker)}
    disabled={selected.length >= 4 && !selected.includes(ticker)} onChange={() => toggle(ticker)} />;
  const series = useDailySparks(sparks && visible("spark") ? rows.map((r) => r.ticker) : []);
  return (
    <div className={sparks && table?.compact ? "scanner-results is-compact" : "scanner-results"}>
      {sparks && prefs.data && <TablePreferences key={JSON.stringify(table)} initial={table ?? { columns: [...DEFAULT_COLUMNS], compact: false }} premium={premium} />}
      {sparks && selected.length > 0 && <div className="row-actions compare-actions">
        <span className="cap">Select 2–4 stocks to compare · {selected.length} selected</span>
        {selected.length >= 2 && <Link className="btn btn-sm" href={`/stocks/compare?symbols=${encodeURIComponent(selected.join(","))}`}>Compare selected</Link>}
        {selected.length > 0 && <button type="button" className="btn btn-sm" onClick={() => setSelected([])}>Clear selection</button>}
      </div>}
      <p className="table-scroll-hint cap">Scroll the table for more columns · tick boxes select 2–4 stocks to compare.</p>
      <div className="table-wrap only-wide" tabIndex={0} role="region" aria-label="Ranked setups table; scroll for more columns">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">{numbered ? "#" : <span className="sr-only">Row</span>}</th><th scope="col">Ticker</th><th scope="col" hidden={!visible("setup")}>Setup</th>
              <th scope="col" className="w30">HSF Score</th>{sparks && <th scope="col" hidden={!visible("spark")}>20 sessions</th>}<th scope="col" hidden={!visible("last")} className="num">Last</th>
              <th scope="col" hidden={!visible("change")} className="num">Chg %</th><th scope="col" hidden={!visible("gap")} className="num">Gap %</th>
              <th scope="col" hidden={!visible("rvol")}>RVOL</th>{premium && <th scope="col" hidden={!visible("prob")} className="num">PreBreakout</th>}
              <th scope="col"><span className="sr-only">Save</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.ticker}-${i}`}>
                <td className="mono cap">{numbered ? offset + i + 1 : ""}</td>
                <td>{sparks && choose(r.ticker)} <TickerLink ticker={r.ticker} />{r.fading && <> <Pill tone="warn">Fading</Pill></>}</td>
                <td hidden={!visible("setup")}><Pill>{setupLabel(r.primary_setup)}</Pill></td>
                <td><ScoreBar score={r.score} /></td>
                {sparks && <td hidden={!visible("spark")}><DailySpark points={series[r.ticker]} ticker={r.ticker} /></td>}
                <td hidden={!visible("last")} className="num mono">{price(r.last)}</td>
                <td hidden={!visible("change")} className={`num mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</td>
                <td hidden={!visible("gap")} className="num mono">{pct(r.gap_pct)}</td>
                <td hidden={!visible("rvol")}><RvolMeter rvol={r.rvol} /></td>
                {premium && <td hidden={!visible("prob")} className="num mono">{probWithRank(r.prob, r.prob_rank)}</td>}
                <td className="num"><SaveToWatchlistButton ticker={r.ticker} compact /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ol className="cards only-narrow" start={offset + 1}>
        {rows.map((r, i) => (
          <li key={`${r.ticker}-${i}`} className="result-card setup-card">
            <div className="sc-top">
              {numbered && <span className="mono cap">#{offset + i + 1}</span>}
              {sparks && choose(r.ticker)}
              <TickerLink ticker={r.ticker} />
              {visible("setup") && <Pill>{setupLabel(r.primary_setup)}</Pill>}
              {r.fading && <Pill tone="warn">Fading</Pill>}
              <span className="grow" />
              {sparks && visible("spark") && <DailySpark points={series[r.ticker]} ticker={r.ticker} width={72} height={22} />}
              <SaveToWatchlistButton ticker={r.ticker} compact />
            </div>
            <dl className="sc-facts">
              <div><dt>HSF</dt><dd><ScoreBadge score={r.score} /></dd></div>
              <div hidden={!visible("last")}><dt>Last</dt><dd className="mono">{price(r.last)}</dd></div>
              <div hidden={!visible("change")}><dt>Chg</dt><dd className={`mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</dd></div>
              <div hidden={!visible("gap")}><dt>Gap</dt><dd className="mono">{pct(r.gap_pct)}</dd></div>
              <div hidden={!visible("rvol")}><dt>RVOL</dt><dd className="mono">{num(r.rvol)}</dd></div>
              {premium && <div hidden={!visible("prob")}><dt>PreBreakout</dt><dd className="mono">{probWithRank(r.prob, r.prob_rank)}</dd></div>}
            </dl>
          </li>
        ))}
      </ol>
      <ResearchGlossary />
      {!premium && <p className="cap table-note">PreBreakout probability is part of <Link href="/pricing">Premium</Link>.</p>}
    </div>
  );
}
