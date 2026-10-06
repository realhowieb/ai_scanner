"use client";

import { useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import { TICKER_RE } from "@/components/AppShell";
import { Card, Disclaimer, Empty, ErrorState, UpgradeButton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useScanJob } from "@/hooks/useScanJob";
import type { ScanApi, ScanCreate, ScanJob } from "@/hooks/useScanJob";
import { useSession } from "@/session/SessionProvider";

import { SetupTable } from "./SetupTable";

type Universe = ScanCreate["universe"];
type Session = "regular" | "premarket" | "afterhours";

export const UNIVERSES: { id: Universe; label: string; feature?: string; plan?: "pro" | "premium" }[] = [
  { id: "sp500", label: "S&P 500" },
  { id: "nasdaq", label: "NASDAQ", feature: "can_scan_nasdaq", plan: "pro" },
  { id: "combo", label: "Combo (S&P 500 + NASDAQ)", feature: "can_scan_nasdaq", plan: "pro" },
  { id: "us_market", label: "US market", feature: "can_full_universe", plan: "premium" },
  { id: "watchlist", label: "A watchlist" },
  { id: "ticker", label: "One ticker" },
];
const SESSIONS: { id: Session; label: string; feature?: string }[] = [
  { id: "regular", label: "Regular" },
  { id: "premarket", label: "Pre-market", feature: "can_premarket" },
  { id: "afterhours", label: "After hours", feature: "can_afterhours" },
];
const TOP_N = [10, 25, 50, 100, 200];
const PHASE_LABEL: Record<string, string> = {
  queued: "Queued, waiting for a scanner",
  starting: "Starting",
  loading_universe: "Loading the stock list",
  scanning: "Scanning",
  finishing: "Ranking results",
  complete: "Complete",
  failed: "Failed",
};

function Progress({ job }: { job: ScanJob }) {
  const p = job.progress;
  const phase = p?.phase ?? job.status;
  return (
    <div className="progress" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <div>
        <p className="strong">{PHASE_LABEL[phase] ?? phase}{p?.symbols ? ` ${p.symbols.toLocaleString()} stocks` : ""}</p>
        <p className="cap">{p?.elapsed_s ? `${Math.round(p.elapsed_s)}s elapsed · ` : ""}You can leave this page; the scan keeps running and reopens here.</p>
      </div>
    </div>
  );
}

function optNum(v: string): number | undefined {
  const n = Number(v);
  return v.trim() === "" || !Number.isFinite(n) ? undefined : n;
}

export function CustomScanView({ client }: { client?: ScanApi } = {}) {
  const { can } = useSession();
  const scan = useScanJob({ client });
  const [universe, setUniverse] = useState<Universe>("sp500");
  const [ticker, setTicker] = useState("");
  const [watchlistId, setWatchlistId] = useState<string>("");
  const [session, setSession] = useState<Session>("regular");
  const [topN, setTopN] = useState(25);
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minDollarVol, setMinDollarVol] = useState("");
  const [unusual, setUnusual] = useState(false);
  const [gap, setGap] = useState(false);
  const [minGap, setMinGap] = useState("");
  const [scoreAll, setScoreAll] = useState(false);
  const [formErr, setFormErr] = useState<string | null>(null);

  // The plan's row cap comes from the server (same cap as the Scanner).
  const cap = useApi("cap", (signal) => unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: 1 } }, signal })));
  const maxRows = cap.data?.max_results ?? 25;
  const lists = useApi(universe === "watchlist" ? "watchlists" : null, (signal) => unwrap(api.GET("/v1/watchlists", { signal })));

  const job = scan.job;
  const active = !!job && (job.status === "queued" || job.status === "running");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    setFormErr(null);
    const t = ticker.trim().toUpperCase();
    if (universe === "ticker" && !TICKER_RE.test(t)) return setFormErr("Enter a ticker symbol like AAPL.");
    if (universe === "watchlist" && !watchlistId) return setFormErr("Choose a watchlist.");
    const filters: NonNullable<ScanCreate["filters"]> = {
      top_n: Math.min(topN, maxRows),
      session,
      ...(optNum(minPrice) !== undefined ? { min_price: optNum(minPrice) } : {}),
      ...(optNum(maxPrice) !== undefined ? { max_price: optNum(maxPrice) } : {}),
      ...(optNum(minDollarVol) !== undefined ? { min_dollar_vol: optNum(minDollarVol) } : {}),
      ...(unusual ? { unusual_volume: true } : {}),
      ...(gap ? { apply_gap_filter: true, ...(optNum(minGap) !== undefined ? { min_gap: optNum(minGap) } : {}) } : {}),
    };
    void scan.start({
      universe,
      ...(universe === "ticker" ? { ticker: t } : {}),
      ...(universe === "watchlist" ? { watchlist_id: Number(watchlistId) } : {}),
      ...((universe === "ticker" || universe === "watchlist") && scoreAll ? { score_all: true } : {}),
      filters,
    });
  };

  const err = scan.error;
  const planErr = err?.status === 403;

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Custom scan</h1>
          <p className="cap">Run the HSF scanner on the list you choose. Large lists take a few minutes.</p>
        </div>
      </section>

      <div className="split">
        <form className="card col-side form" onSubmit={submit} aria-label="Scan settings">
          <fieldset disabled={active || scan.starting}>
            <legend className="h2">What to scan</legend>
            <div className="radio-list">
              {UNIVERSES.map((u) => {
                const locked = !!u.feature && !can(u.feature);
                return (
                  <label key={u.id} className={`radio${locked ? " is-locked" : ""}`}>
                    <input type="radio" name="universe" value={u.id} checked={universe === u.id} disabled={locked}
                      onChange={() => setUniverse(u.id)} />
                    <span>{u.label}</span>
                    {locked && <span className="pill pill-gold">{u.plan === "premium" ? "Premium" : "Pro"}</span>}
                  </label>
                );
              })}
            </div>
            {can("can_full_universe") && (universe === "nasdaq" || universe === "combo") && (
              <p className="cap">Premium scans the full list.</p>
            )}
            {universe === "ticker" && (
              <label className="field"><span>Ticker</span>
                <input value={ticker} maxLength={10} autoComplete="off" onChange={(e) => setTicker(e.target.value)} placeholder="AAPL" />
              </label>
            )}
            {universe === "watchlist" && (
              <label className="field"><span>Watchlist</span>
                <select value={watchlistId} onChange={(e) => setWatchlistId(e.target.value)}>
                  <option value="">{lists.loading ? "Loading…" : lists.data?.length ? "Choose…" : "No watchlists yet"}</option>
                  {(lists.data ?? []).map((w) => <option key={w.id} value={w.id}>{w.name} ({w.symbol_count})</option>)}
                </select>
              </label>
            )}
            {(universe === "ticker" || universe === "watchlist") && (
              <label className="check"><input type="checkbox" checked={scoreAll} onChange={(e) => setScoreAll(e.target.checked)} />
                <span>Score every ticker, not only setups</span></label>
            )}

            <legend className="h2 mt">Session</legend>
            <div className="seg-group" role="radiogroup" aria-label="Session">
              {SESSIONS.map((s) => {
                const locked = !!s.feature && !can(s.feature);
                return (
                  <label key={s.id} className={`seg${session === s.id ? " on" : ""}${locked ? " is-locked" : ""}`}>
                    <input type="radio" name="session" className="sr-only" checked={session === s.id} disabled={locked} onChange={() => setSession(s.id)} />
                    {s.label}{locked ? " · Pro" : ""}
                  </label>
                );
              })}
            </div>

            <legend className="h2 mt">Filters</legend>
            <div className="grid2">
              <label className="field"><span>Min price ($)</span><input inputMode="decimal" value={minPrice} onChange={(e) => setMinPrice(e.target.value)} placeholder="Default" /></label>
              <label className="field"><span>Max price ($)</span><input inputMode="decimal" value={maxPrice} onChange={(e) => setMaxPrice(e.target.value)} placeholder="Default" /></label>
              <label className="field"><span>Min dollar volume</span><input inputMode="numeric" value={minDollarVol} onChange={(e) => setMinDollarVol(e.target.value)} placeholder="Default" /></label>
              <label className="field"><span>Results</span>
                <select value={Math.min(topN, maxRows)} onChange={(e) => setTopN(Number(e.target.value))}>
                  {TOP_N.map((n) => <option key={n} value={n} disabled={n > maxRows}>{n}{n > maxRows ? " (upgrade)" : ""}</option>)}
                </select>
              </label>
            </div>
            <label className={`check${can("can_unusual_volume") ? "" : " is-locked"}`}>
              <input type="checkbox" checked={unusual} disabled={!can("can_unusual_volume")} onChange={(e) => setUnusual(e.target.checked)} />
              <span>Unusual volume only{can("can_unusual_volume") ? "" : " · Pro"}</span>
            </label>
            <label className={`check${can("can_scan_nasdaq") ? "" : " is-locked"}`}>
              <input type="checkbox" checked={gap} disabled={!can("can_scan_nasdaq")} onChange={(e) => setGap(e.target.checked)} />
              <span>Gap filter{can("can_scan_nasdaq") ? "" : " · Pro"}</span>
            </label>
            {gap && <label className="field"><span>Min gap (%)</span><input inputMode="decimal" value={minGap} onChange={(e) => setMinGap(e.target.value)} placeholder="Default" /></label>}
          </fieldset>
          {formErr && <p className="form-error" role="alert">{formErr}</p>}
          <button type="submit" className="btn btn-primary btn-block" disabled={active || scan.starting || !!scan.retryInS}>
            {scan.starting ? "Starting…" : active ? "Scan running…" : scan.retryInS ? `Try again in ${scan.retryInS}s` : "Start scan"}
          </button>
        </form>

        <div className="col-main stack">
          {err && !(err.status === 409 && job) && (
            planErr ? (
              <Card><div className="locked"><p className="strong">{err.message}</p><UpgradeButton plan={/premium/i.test(err.message) ? "premium" : "pro"} /></div></Card>
            ) : err.status === 429 ? (
              <p className="form-error" role="alert">{err.message} {scan.retryInS ? `You can start another in ${scan.retryInS}s.` : ""}</p>
            ) : err.status === 503 ? (
              <p className="form-error" role="alert">The scanner is busy right now. {scan.retryInS ? `Try again in ${scan.retryInS}s.` : "Try again in a minute."}</p>
            ) : (
              <ErrorState error={err} what="the scan" />
            )
          )}
          {err?.status === 409 && job && <p className="notice" role="status">You already had a scan running. Showing that one.</p>}

          {!job && !err && (
            <Card><Empty title="Choose a list and start a scan.">Results are ranked by HSF Score and saved to your scan history.</Empty></Card>
          )}
          {job && active && <Card title="Scan in progress"><Progress job={job} /></Card>}
          {job && job.status === "failed" && (
            <Card title="The scan didn't finish">
              <p role="alert">{job.error || "The scan failed."}</p>
              <button type="button" className="btn" onClick={scan.dismiss}>Start a new scan</button>
            </Card>
          )}
          {job && job.status === "complete" && job.result && (
            <Card title={`Results · ${job.result.label}`} aside={`${job.result.symbols_scanned.toLocaleString()} stocks scanned in ${Math.round(job.result.duration_s)}s`}>
              {job.result.setups.length === 0 ? (
                <Empty title="No setups matched.">Try a larger list or looser filters.</Empty>
              ) : (
                <div className="flush-inner"><SetupTable rows={job.result.setups} premium={can("can_early_breakout")} /></div>
              )}
              <div className="card-foot">
                <span className="cap">{job.params.session !== job.params.session_requested ? "Outside that session, the regular session was scanned. " : ""}Prices are from the scan.</span>
                <button type="button" className="btn" onClick={scan.dismiss}>New scan</button>
              </div>
            </Card>
          )}
          <Disclaimer />
        </div>
      </div>
    </div>
  );
}
