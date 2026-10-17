"use client";

// Compare stocks: up to three tickers side by side, from the same Stock Intelligence each
// stock page shows (GET /v1/stocks/{ticker}). The tickers live in the URL
// (?tickers=AAPL,MSFT) so a comparison can be bookmarked or shared.
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { TICKER_RE } from "@/components/AppShell";
import { Card, Empty, Pill, ResearchNotice, ScoreBadge, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime, num, pct, price, probPct, setupLabel } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

type Stock = Schemas["StockDetail"];
export const MAX_COMPARE = 3;

export function parseCompare(raw: string | null): string[] {
  const out: string[] = [];
  for (const t of (raw ?? "").split(/[\s,]+/)) {
    const s = t.trim().toUpperCase();
    if (s && TICKER_RE.test(s) && !out.includes(s)) out.push(s);
    if (out.length >= MAX_COMPARE) break;
  }
  return out;
}

/** Percent change of the last daily close vs the close `sessions` bars earlier. */
export function barChange(bars: Stock["bars"], sessions: number): number | null {
  const closes = bars.map((b) => b.close).filter((c): c is number => typeof c === "number" && c > 0);
  if (closes.length <= sessions) return null;
  const now = closes[closes.length - 1]!, then = closes[closes.length - 1 - sessions]!;
  return ((now - then) / then) * 100;
}

const tone = (v: number | null | undefined) => ((v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "");
type Result = { ticker: string; stock: Stock | null; error: string | null };

export function CompareView() {
  const params = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const tickers = parseCompare(params.get("tickers"));
  const [entry, setEntry] = useState("");
  const [bad, setBad] = useState<string | null>(null);
  const { can } = useSession();
  const premium = can("can_early_breakout");
  const key = tickers.join(",");
  const res = useApi(tickers.length ? `compare:${key}` : null, async (signal) => {
    const settled = await Promise.allSettled(tickers.map((t) =>
      unwrap(api.GET("/v1/stocks/{ticker}", { params: { path: { ticker: t } }, signal }))));
    return settled.map((r, i): Result => r.status === "fulfilled"
      ? { ticker: tickers[i]!, stock: r.value, error: null }
      : { ticker: tickers[i]!, stock: null, error: r.reason instanceof Error ? r.reason.message : "Couldn't load." });
  });

  const go = (next: string[]) => router.replace(next.length ? `${path}?tickers=${next.join(",")}` : path, { scroll: false });
  const add = (e: FormEvent) => {
    e.preventDefault();
    const t = entry.trim().toUpperCase().replace(/^\$/, "");
    if (!TICKER_RE.test(t)) { setBad("Enter a ticker symbol like AAPL."); return; }
    setBad(null);
    setEntry("");
    if (!tickers.includes(t)) go([...tickers, t].slice(0, MAX_COMPARE));
  };

  const cols = res.data ?? [];
  const row = (label: string, cell: (s: Stock) => ReactNode) => (
    <tr>
      <th scope="row">{label}</th>
      {cols.map((c) => <td key={c.ticker}>{c.stock ? cell(c.stock) : "—"}</td>)}
    </tr>
  );

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Compare stocks</h1>
          <p className="cap">Up to {MAX_COMPARE} tickers side by side: HSF Score, setup, momentum, volume and earnings, from the latest market scan.</p>
        </div>
      </section>

      <Card title="Tickers" id="cmp-pick">
        {tickers.length > 0 && (
          <div className="chips">
            {tickers.map((t) => (
              <span key={t} className="pill">
                <span className="mono">{t}</span>{" "}
                <button type="button" className="link-btn" aria-label={`Remove ${t}`} onClick={() => go(tickers.filter((x) => x !== t))}>×</button>
              </span>
            ))}
          </div>
        )}
        {tickers.length < MAX_COMPARE ? (
          <form className="inline-form" onSubmit={add}>
            <label className="field"><span>Add a ticker</span>
              <input value={entry} onChange={(e) => setEntry(e.target.value.toUpperCase())} maxLength={10} placeholder="MSFT"
                autoCapitalize="characters" spellCheck={false} />
            </label>
            <button type="submit" className="btn">Add</button>
          </form>
        ) : <p className="cap">Remove one to compare a different stock.</p>}
        {bad && <p className="error" role="alert">{bad}</p>}
      </Card>

      {tickers.length === 0 ? (
        <Card><Empty title="Pick stocks to compare.">Add two or three tickers above, or use Compare on a stock page.</Empty></Card>
      ) : !res.data ? <Skeleton rows={8} label="Loading the comparison" /> : (
        <Card title="Side by side" id="cmp" aside={cols.find((c) => c.stock?.scan_at)?.stock?.scan_at ? `Scan ${etTime(cols.find((c) => c.stock?.scan_at)!.stock!.scan_at)}` : undefined}>
          <div className="table-wrap">
            <table className="table compare-table">
              <thead>
                <tr>
                  <th scope="col"><span className="sr-only">Measure</span></th>
                  {cols.map((c) => <th key={c.ticker} scope="col"><Link href={`/stocks/${encodeURIComponent(c.ticker)}`} className="mono strong">{c.ticker}</Link></th>)}
                </tr>
              </thead>
              <tbody>
                {cols.some((c) => c.error) && (
                  <tr><th scope="row">Status</th>{cols.map((c) => <td key={c.ticker} className="cap">{c.error ?? ""}</td>)}</tr>
                )}
                {row("HSF Score", (s) => s.hsf_score == null ? <span className="cap">Not ranked</span> : <ScoreBadge score={s.hsf_score} />)}
                {row("Change since last scan", (s) => s.score_change ? <span className={`mono ${tone(s.score_change)}`}>{s.score_change > 0 ? "+" : ""}{s.score_change}</span> : "—")}
                {row("Setup", (s) => s.primary_setup ? setupLabel(s.primary_setup) : "—")}
                {row("Status", (s) => s.status ? <Pill tone={s.status === "STRONG" ? "up" : s.status === "CAUTION" || s.status === "FADING" ? "warn" : undefined}>{s.status}</Pill> : "—")}
                {row("Price (scan)", (s) => <span className="mono">{price(s.price)}</span>)}
                {row("Day change", (s) => <span className={`mono ${tone(s.change_pct)}`}>{pct(s.change_pct)}</span>)}
                {row("1 month", (s) => { const v = barChange(s.bars, 21); return <span className={`mono ${tone(v)}`}>{pct(v)}</span>; })}
                {row("1 year", (s) => { const v = barChange(s.bars, 250); return <span className={`mono ${tone(v)}`}>{pct(v)}</span>; })}
                {row("Relative volume", (s) => s.rvol != null ? <span className="mono">{num(s.rvol)}x</span> : "—")}
                {row("Breakout score", (s) => s.breakout_score != null && Number.isFinite(s.breakout_score) ? <span className="mono">{s.breakout_score.toFixed(1)}</span> : "—")}
                {row("PreBreakout", (s) => premium ? <span className="mono">{probPct(s.prob)}</span> : <Pill tone="gold">Premium</Pill>)}
                {row("Signals", (s) => s.signals.length ? s.signals.map(setupLabel).join(", ") : "—")}
                {row("Earnings", (s) => s.earnings_days == null ? "—" : s.earnings_days === 0 ? <Pill tone="warn">Today</Pill> : `in ${s.earnings_days}d`)}
                {row("Main reason", (s) => <span className="body-sm">{s.reasons[0] ?? "—"}</span>)}
                {row("Top risk", (s) => <span className="body-sm">{s.risks[0] ?? "—"}</span>)}
                {row("Your watchlists", (s) => s.watchlists.length ? s.watchlists.map((w) => w.name).join(", ") : "—")}
              </tbody>
            </table>
          </div>
          <ResearchNotice compact />
        </Card>
      )}
    </div>
  );
}
