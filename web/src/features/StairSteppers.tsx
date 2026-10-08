"use client";

// Day Trader's stair-stepper check (Pro): which of the symbols on screen are moving in a
// tight, straight line on the 1-minute chart, from /v1/day-trader/stair-steppers. Runs
// on request. Descriptive only; the thresholds are the classic app's defaults.
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, Empty, ErrorLine, TickerLink } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { etTime, num } from "@/lib/format";

export const MAX_CHECK = 40;
const WINDOWS = [10, 15, 20, 30, 45, 60] as const;
type Direction = "up" | "down" | "either";
type Row = { ticker: string; status?: string; r2?: number | null; trend_pct_per_hour?: number | null;
  max_pullback_pct?: number | null; bars?: number; as_of?: string | null };

export function StairSteppers({ symbols }: { symbols: string[] }) {
  const [direction, setDirection] = useState<Direction>("up");
  const [window, setWindow] = useState<number>(45);
  const [r2, setR2] = useState(0.8);
  const [pullback, setPullback] = useState(1.0);
  const [trend, setTrend] = useState(0.5);
  const [out, setOut] = useState<Schemas["StairSteppers"] | null>(null);
  const act = useAction();
  const checked = symbols.slice(0, MAX_CHECK);
  const run = () => void act.run(async () => {
    setOut(await unwrap(api.GET("/v1/day-trader/stair-steppers", { params: { query: {
      symbols: checked.join(","), window, direction, r2_min: r2, max_pullback: pullback, min_trend: trend,
    } } })));
    return true;
  });
  const matches = ((out?.matches ?? []) as Row[]).slice().sort((a, b) => (b.r2 ?? 0) - (a.r2 ?? 0));
  const all = (out?.all ?? []) as Row[];
  const thin = all.filter((r) => r.status !== "ok").map((r) => r.ticker);
  const asOf = all.map((r) => r.as_of).filter(Boolean).sort().pop();
  return (
    <Card title="Stair-steppers" id="stair" aside="Smooth 1-minute trends">
      <p className="cap">Finds symbols moving in a tight, straight line on the 1-minute chart. R² is how closely the last bars follow a straight line (1.0 = perfectly straight). It describes the recent path; it is not a prediction.</p>
      <div className="form-grid">
        <label className="field"><span>Direction</span>
          <select value={direction} onChange={(e) => setDirection(e.target.value as Direction)}>
            <option value="up">Up</option><option value="down">Down</option><option value="either">Either</option>
          </select>
        </label>
        <label className="field"><span>Bars fitted</span>
          <select value={window} onChange={(e) => setWindow(Number(e.target.value))}>
            {WINDOWS.map((w) => <option key={w} value={w}>{w} one-minute bars</option>)}
          </select>
        </label>
        <label className="field"><span>Minimum R²</span>
          <input type="number" inputMode="decimal" min={0.5} max={0.99} step={0.01} value={r2} onChange={(e) => setR2(Number(e.target.value))} />
        </label>
        <label className="field"><span>Max pullback %</span>
          <input type="number" inputMode="decimal" min={0.1} max={5} step={0.1} value={pullback} onChange={(e) => setPullback(Number(e.target.value))} />
        </label>
        <label className="field"><span>Min trend % per hour</span>
          <input type="number" inputMode="decimal" min={0} max={20} step={0.1} value={trend} onChange={(e) => setTrend(Number(e.target.value))} />
        </label>
      </div>
      <div className="row-actions">
        <button type="button" className="btn btn-primary" disabled={act.busy || checked.length === 0} onClick={run}>
          {act.busy ? "Loading 1-minute bars…" : `Check ${checked.length} symbol${checked.length === 1 ? "" : "s"}`}
        </button>
        {symbols.length > MAX_CHECK && <span className="cap">Checks the top {MAX_CHECK} of {symbols.length} by DT score.</span>}
      </div>
      <ErrorLine error={act.error} />
      {out && (
        <>
          {asOf && <p className="cap">Latest 1-minute bar: {etTime(asOf)}.</p>}
          {matches.length === 0 ? <Empty title="No symbols match these settings right now." /> : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Ticker</th><th scope="col" className="num">R²</th><th scope="col" className="num">Trend %/hr</th>
                  <th scope="col" className="num">Max pullback %</th><th scope="col" className="num hide-narrow">Bars</th></tr></thead>
                <tbody>
                  {matches.map((r) => (
                    <tr key={r.ticker}>
                      <td><TickerLink ticker={r.ticker} /></td>
                      <td className="num mono strong">{r.r2 == null ? "—" : r.r2.toFixed(3)}</td>
                      <td className={`num mono ${(r.trend_pct_per_hour ?? 0) >= 0 ? "up" : "down"}`}>{num(r.trend_pct_per_hour, 2)}</td>
                      <td className="num mono">{num(r.max_pullback_pct, 2)}</td>
                      <td className="num mono hide-narrow">{r.bars ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {thin.length > 0 && <p className="cap">Not enough 1-minute data to judge ({thin.length}): {thin.slice(0, 15).join(", ")}{thin.length > 15 ? "…" : ""}. Thinly traded names have gaps on the free data feed.</p>}
        </>
      )}
    </Card>
  );
}
