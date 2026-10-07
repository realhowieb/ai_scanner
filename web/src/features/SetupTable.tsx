"use client";

import type { Schemas } from "@/api/client";
import { Pill, ScoreBar, TickerLink } from "@/components/ui";
import { num, pct, price, probPct, setupLabel } from "@/lib/format";

import { SaveToWatchlistButton } from "./SaveToWatchlist";

type Row = Schemas["ScanSetup"];

function prob(r: Row, premium: boolean) {
  if (!premium) return <span className="cap" title="PreBreakout probability is part of Premium">Premium</span>;
  return probPct(r.prob);
}

/** Ranked rows, as a table on wide screens and as cards on phones. Rows come ranked from the API. */
export function SetupTable({ rows, offset = 0, premium }: { rows: Row[]; offset?: number; premium: boolean }) {
  return (
    <>
      <div className="table-wrap only-wide">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">#</th><th scope="col">Ticker</th><th scope="col">Setup</th>
              <th scope="col" className="w30">HSF Score</th><th scope="col" className="num">Last</th>
              <th scope="col" className="num">Chg %</th><th scope="col" className="num">Gap %</th>
              <th scope="col" className="num">RVOL</th><th scope="col" className="num">PreBreakout</th>
              <th scope="col"><span className="sr-only">Save</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.ticker}-${i}`}>
                <td className="mono cap">{offset + i + 1}</td>
                <td><TickerLink ticker={r.ticker} />{r.fading && <> <Pill tone="warn">Fading</Pill></>}</td>
                <td><Pill>{setupLabel(r.primary_setup)}</Pill></td>
                <td><ScoreBar score={r.score} /></td>
                <td className="num mono">{price(r.last)}</td>
                <td className={`num mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</td>
                <td className="num mono">{pct(r.gap_pct)}</td>
                <td className="num mono">{num(r.rvol)}</td>
                <td className="num mono">{prob(r, premium)}</td>
                <td className="num"><SaveToWatchlistButton ticker={r.ticker} compact /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ol className="cards only-narrow" start={offset + 1}>
        {rows.map((r, i) => (
          <li key={`${r.ticker}-${i}`} className="result-card">
            <div className="rc-top">
              <span className="mono cap">#{offset + i + 1}</span>
              <TickerLink ticker={r.ticker} />
              <Pill>{setupLabel(r.primary_setup)}</Pill>
              {r.fading && <Pill tone="warn">Fading</Pill>}
              <span className="grow" />
              <SaveToWatchlistButton ticker={r.ticker} compact />
            </div>
            <ScoreBar score={r.score} />
            <dl className="rc-facts">
              <div><dt>Last</dt><dd className="mono">{price(r.last)}</dd></div>
              <div><dt>Chg</dt><dd className={`mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</dd></div>
              <div><dt>RVOL</dt><dd className="mono">{num(r.rvol)}</dd></div>
              <div><dt>PreBreakout</dt><dd className="mono">{prob(r, premium)}</dd></div>
            </dl>
          </li>
        ))}
      </ol>
    </>
  );
}
