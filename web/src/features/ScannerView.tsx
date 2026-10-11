"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { DataFreshness } from "@/features/DataFreshness";
import { api, unwrap } from "@/api/client";
import { Card, Disclaimer, Empty, ErrorState, Freshness, Locked, ResearchNotice, Skeleton, UpgradeButton } from "@/components/ui";
import { AIScanPanel } from "@/features/AIScanPanel";
import { useApi } from "@/hooks/useApi";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { etTime, freshness } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

import { SetupTable } from "./SetupTable";

export const SIGNALS = [
  { id: "", label: "All setups" },
  { id: "breakout", label: "Breakout" },
  { id: "gapper", label: "Gapper" },
  { id: "golden_cross", label: "Golden cross" },
  { id: "gainer", label: "Gainer" },
  { id: "prebreakout", label: "PreBreakout", feature: "can_early_breakout" },
] as const;
const MIN_SCORES = [0, 40, 60, 75];
const PAGE_SIZES = [25, 50, 100];
export const SORTS = [
  { id: "score", label: "HSF Score" },
  { id: "chg_pct", label: "Change %" },
  { id: "gap_pct", label: "Gap %" },
  { id: "rvol", label: "Relative volume" },
  { id: "prob", label: "PreBreakout", feature: "can_early_breakout" },
] as const;
type Sort = (typeof SORTS)[number]["id"];

type Signal = "golden_cross" | "breakout" | "prebreakout" | "gapper" | "gainer";

export function readFilters(sp: URLSearchParams) {
  const sig = sp.get("signal") || "";
  const signal = (SIGNALS.some((s) => s.id === sig) ? sig : "") as Signal | "";
  const min = Number(sp.get("min") || 0);
  const size = Number(sp.get("size") || 25);
  const page = Math.max(1, Math.floor(Number(sp.get("page") || 1)) || 1);
  const so = sp.get("sort") || "score";
  return {
    sort: (SORTS.some((x) => x.id === so) ? so : "score") as Sort,
    signal,
    minScore: MIN_SCORES.includes(min) ? min : 0,
    size: PAGE_SIZES.includes(size) ? size : 25,
    page,
  };
}

export function ScannerView() {
  const sp = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const { can } = useSession();
  const f = readFilters(new URLSearchParams(sp.toString()));
  const premium = can("can_early_breakout");
  const lockedSignal = f.signal === "prebreakout" && !premium;
  const offset = (f.page - 1) * f.size;

  const sort: Sort = f.sort === "prob" && !premium ? "score" : f.sort;
  const key = lockedSignal ? null : `scan:${f.signal}:${f.minScore}:${f.size}:${f.page}:${sort}`;
  const { data, error, loading, reload } = useApi(key, (signal) =>
    unwrap(api.GET("/v1/scans/latest", {
      params: { query: { limit: f.size, offset, min_score: f.minScore, ...(f.signal ? { signal: f.signal } : {}), ...(sort !== "score" ? { sort } : {}) } },
      signal,
    })));
  useAutoRefresh(reload, { enabled: key !== null });

  const set = (patch: Record<string, string | number>) => {
    const next = new URLSearchParams(sp.toString());
    for (const [k, v] of Object.entries(patch)) {
      if (v === "" || v === 0 || (k === "page" && v === 1) || (k === "size" && v === 25) || (k === "sort" && v === "score")) next.delete(k);
      else next.set(k, String(v));
    }
    if (!("page" in patch)) next.delete("page");
    const qs = next.toString();
    router.replace(qs ? `${path}?${qs}` : path, { scroll: false });
  };

  const visibleTotal = data ? Math.min(data.total, data.max_results) : 0;
  const pages = Math.max(1, Math.ceil(visibleTotal / f.size));
  const filtersActive = Boolean(f.signal || f.minScore || f.size !== 25 || sort !== "score");
  const leader = data?.setups[0];
  const setupCountLabel = data ? `${data.total} setup${data.total === 1 ? "" : "s"} ranked` : "Loading the latest ranked setups";

  return (
    <div className="stack">
      <DataFreshness info={data?.freshness} />
      <section className="page-head">
        <div>
          <h1 className="h1">Scanner</h1>
          <p className="cap">
            {data?.scan_at ? <Freshness at={data.scan_at} label="Latest full-market scan" stale={data.stale} /> : "Latest full-market scan"}
            {data && ` · ${data.total} setup${data.total === 1 ? "" : "s"} ranked`}
          </p>
        </div>
        <div className="row-actions">
          <Link href="/scanner/history" className="btn">Scan history</Link>
          <Link href="/scanner/custom" className="btn btn-primary">Run custom scan</Link>
        </div>
      </section>

      <section className="priority-panel scanner-priority" aria-labelledby="scanner-priority-title">
        <div className="priority-copy">
          <p className="cap">Best starting point</p>
          <h2 className="h2" id="scanner-priority-title">
            {leader ? `${leader.ticker} leads the latest HSF-ranked scan.` : setupCountLabel}
          </h2>
          <p className="body-sm">
            {leader
              ? `Review the score evidence first, then refine by setup type, score floor, or session context when you need a narrower list.`
              : "Start with the ranked board, then narrow the list only when you know what kind of setup you want."}
          </p>
        </div>
        <ResearchNotice compact />
      </section>

      <details className="filter-panel" open={filtersActive}>
        <summary>
          <span className="strong">Refine results</span>
          <span className="cap">{filtersActive ? "Filters active" : "Setup type, score floor, sort, and rows"}</span>
        </summary>
        <section aria-label="Filters" className="filters">
          <div className="chips" role="group" aria-label="Setup type">
            {SIGNALS.map((s) => {
              const locked = "feature" in s && !can(s.feature);
              return (
                <button key={s.id || "all"} type="button" className="chip" aria-pressed={f.signal === s.id}
                  onClick={() => set({ signal: s.id })}>
                  {s.label}{locked && <span className="chip-lock"> · Premium</span>}
                </button>
              );
            })}
          </div>
          <span className="grow" />
          <label className="inline-field">Min HSF Score
            <select value={f.minScore} onChange={(e) => set({ min: Number(e.target.value) })}>
              {MIN_SCORES.map((m) => <option key={m} value={m}>{m === 0 ? "Any" : m}</option>)}
            </select>
          </label>
          <label className="inline-field">Sort by
            <select value={sort} onChange={(e) => set({ sort: e.target.value })}>
              {SORTS.filter((x) => !("feature" in x) || can(x.feature)).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
          </label>
          <label className="inline-field">Rows
            <select value={f.size} onChange={(e) => set({ size: Number(e.target.value) })}>
              {PAGE_SIZES.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
        </section>
      </details>

      {data?.scan_at && (data.stale ?? freshness(data.scan_at).stale) && (
        <p className="banner" role="status">Market data is delayed: a scheduled scan didn&apos;t arrive, so this is the scan from {etTime(data.scan_at)}. Scores and prices are from that scan.</p>
      )}

      {lockedSignal ? (
        <Card><Locked title="PreBreakout is part of Premium" plan="premium">The PreBreakout model ranks names before they break out, with its probability.</Locked></Card>
      ) : error && !data ? (
        <ErrorState error={error} onRetry={reload} what="the Scanner" />
      ) : loading && !data ? (
        <Skeleton rows={10} label="Loading scan results" />
      ) : data && !data.scan_at ? (
        <Card><Empty title="No market scan yet.">Results appear after the first scheduled scan.</Empty></Card>
      ) : data && data.setups.length === 0 ? (
        <Card><Empty title={f.signal || f.minScore ? "No setups match these filters." : "The latest scan has no ranked setups."}>
          {(f.signal || f.minScore) ? <button type="button" className="btn" onClick={() => set({ signal: "", min: 0 })}>Clear filters</button> : null}
        </Empty></Card>
      ) : data ? (
        <section className={`card flush${loading ? " dim" : ""}`} aria-busy={loading}>
          <SetupTable rows={data.setups} offset={offset} premium={premium} numbered={sort === "score"} />
        </section>
      ) : null}

      {data && sort !== "score" && <p className="cap">Sorted by {SORTS.find((x) => x.id === sort)!.label.toLowerCase()} within your plan&apos;s top {Math.min(data.total, data.max_results)} HSF-ranked setups.</p>}
      {data && data.limited && (
        <div className="cap-note" role="note">
          <p>Your plan shows the top <b>{data.max_results}</b> of <b>{data.total}</b> setups in this scan.</p>
          <UpgradeButton plan={data.max_results < 100 ? "pro" : "premium"} />
        </div>
      )}

      {data && pages > 1 && (
        <nav className="pager" aria-label="Pages">
          <button type="button" className="btn" disabled={f.page <= 1} onClick={() => set({ page: f.page - 1 })}>Previous</button>
          <span className="cap">Page {f.page} of {pages}</span>
          <button type="button" className="btn" disabled={f.page >= pages} onClick={() => set({ page: f.page + 1 })}>Next</button>
        </nav>
      )}
      {data && data.setups.length > 0 && <AIScanPanel />}
      <Disclaimer />
    </div>
  );
}
