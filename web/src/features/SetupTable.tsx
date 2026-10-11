"use client";

import Link from "next/link";

import type { Schemas } from "@/api/client";
import { Pill, ScoreBadge, ScoreBar, TickerLink } from "@/components/ui";
import { num, pct, price, probWithRank, setupLabel } from "@/lib/format";

import { SaveToWatchlistButton } from "./SaveToWatchlist";

type Row = Schemas["ScanSetup"];

/** Ranked rows, as a table on wide screens and as cards on phones. Rows come ranked from the API. */
/** `numbered` = rows are in HSF rank order, so the position is the rank; off for other sort orders. */
/** The PreBreakout column shows only on Premium; other plans get one note under the rows instead of "Premium" per row. */
export function SetupTable({ rows, offset = 0, premium, numbered = true }: { rows: Row[]; offset?: number; premium: boolean; numbered?: boolean }) {
  return (
    <>
      <div className="table-wrap only-wide">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">{numbered ? "#" : <span className="sr-only">Row</span>}</th><th scope="col">Ticker</th><th scope="col">Setup</th>
              <th scope="col" className="w30">HSF Score</th><th scope="col" className="num">Last</th>
              <th scope="col" className="num">Chg %</th><th scope="col" className="num">Gap %</th>
              <th scope="col" className="num">RVOL</th>{premium && <th scope="col" className="num">PreBreakout</th>}
              <th scope="col"><span className="sr-only">Save</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.ticker}-${i}`}>
                <td className="mono cap">{numbered ? offset + i + 1 : ""}</td>
                <td><TickerLink ticker={r.ticker} />{r.fading && <> <Pill tone="warn">Fading</Pill></>}</td>
                <td><Pill>{setupLabel(r.primary_setup)}</Pill></td>
                <td><ScoreBar score={r.score} /></td>
                <td className="num mono">{price(r.last)}</td>
                <td className={`num mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</td>
                <td className="num mono">{pct(r.gap_pct)}</td>
                <td className="num mono">{num(r.rvol)}</td>
                {premium && <td className="num mono">{probWithRank(r.prob, r.prob_rank)}</td>}
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
              <TickerLink ticker={r.ticker} />
              <Pill>{setupLabel(r.primary_setup)}</Pill>
              {r.fading && <Pill tone="warn">Fading</Pill>}
              <span className="grow" />
              <SaveToWatchlistButton ticker={r.ticker} compact />
            </div>
            <dl className="sc-facts">
              <div><dt>HSF</dt><dd><ScoreBadge score={r.score} /></dd></div>
              <div><dt>Last</dt><dd className="mono">{price(r.last)}</dd></div>
              <div><dt>Chg</dt><dd className={`mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</dd></div>
              <div><dt>RVOL</dt><dd className="mono">{num(r.rvol)}</dd></div>
              {premium && <div><dt>PreBreakout</dt><dd className="mono">{probWithRank(r.prob, r.prob_rank)}</dd></div>}
            </dl>
          </li>
        ))}
      </ol>
      {!premium && <p className="cap table-note">PreBreakout probability is part of <Link href="/pricing">Premium</Link>.</p>}
    </>
  );
}
