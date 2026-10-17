"use client";

// Scan history (Pro): your own saved scans (/v1/runs) and one saved scan's ranked rows
// (/v1/runs/{id}), shown with the same table as the Scanner.
import Link from "next/link";
import { Fragment, useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, Empty, ErrorState, Locked, Skeleton, UpgradeButton } from "@/components/ui";
import { AIScanPanel } from "@/features/AIScanPanel";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

import { SetupTable } from "./SetupTable";

function Head({ title, sub }: { title: string; sub?: string }) {
  return (
    <section className="page-head">
      <div>
        <h1 className="h1">{title}</h1>
        {sub && <p className="cap">{sub}</p>}
      </div>
      <Link href="/scanner/custom" className="btn btn-primary">Run custom scan</Link>
    </section>
  );
}

const LOCKED = <Card><Locked title="Scan history is part of Pro" plan="pro">Every custom scan you run is saved, so you can reopen its ranked results later.</Locked></Card>;

export function ScanHistoryView() {
  const { can } = useSession();
  const allowed = can("can_scan_history");
  const runs = useApi(allowed ? "runs" : null, (signal) => unwrap(api.GET("/v1/runs", { params: { query: { limit: 50 } }, signal })));
  const [query, setQuery] = useState("");
  const [from, setFrom] = useState("");
  const [until, setUntil] = useState("");
  const [universe, setUniverse] = useState("");
  const names = [...new Set((runs.data ?? []).map((r) => r.label || r.name || "Unnamed scan"))];
  const shown = (runs.data ?? []).filter((r) => {
    const label = r.label || r.name || "Unnamed scan";
    const day = new Date(r.created_at ?? "").toLocaleDateString("en-CA", { timeZone: "America/New_York" });
    if ((from || until) && !r.created_at) return false;
    return label.toLowerCase().replace(/[^a-z0-9]/g, "").includes(query.toLowerCase().replace(/[^a-z0-9]/g, "")) && (!universe || label === universe) && (!from || day >= from) && (!until || day <= until);
  });
  return (
    <div className="stack">
      <Head title="Scan history" sub="Your saved custom scans, newest first." />
      {allowed && <div className="filters history-filters">
        <label className="field">Search scans<input type="search" value={query} onChange={(e) => setQuery(e.target.value)} /></label>
        <label className="field">Scan source<select value={universe} onChange={(e) => setUniverse(e.target.value)}><option value="">All sources</option>{names.map((name) => <option key={name} value={name}>{name.replace(/\bSP500\b/g, "S&P 500")}</option>)}</select></label>
        <label className="field">From (ET)<input type="date" value={from} onChange={(e) => setFrom(e.target.value)} /></label>
        <label className="field">Through (ET)<input type="date" min={from || undefined} value={until} onChange={(e) => setUntil(e.target.value)} /></label>
        <button type="button" className="btn btn-sm" onClick={() => { setQuery(""); setFrom(""); setUntil(""); setUniverse(""); }}>Clear filters</button>
        <p className="cap">Showing {shown.length} of the latest {runs.data?.length ?? 0} saved scans (up to 50).</p>
      </div>}
      {!allowed ? LOCKED
        : runs.error && !runs.data ? (runs.error.status === 403 ? LOCKED : <ErrorState error={runs.error} onRetry={runs.reload} what="your scan history" />)
        : !runs.data ? <Skeleton rows={6} label="Loading scan history" />
        : runs.data.length === 0 ? <Card><Empty title="No saved scans yet."><Link href="/scanner/custom">Run a custom scan</Link> and it will be saved here.</Empty></Card>
        : (
          <section className="card flush">
            {shown.length === 0 && <Empty title="No scans match these filters." />}
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Scan</th><th scope="col">Run</th><th scope="col" className="num">Rows</th><th scope="col" className="num hide-narrow">Took</th></tr></thead>
                <tbody>
                  {shown.map((r, i) => (
                    <Fragment key={r.id}>{(i === 0 || etTime(shown[i - 1]?.created_at).split(",")[0] !== etTime(r.created_at).split(",")[0]) && <tr><td colSpan={4} className="history-day">{r.created_at ? new Date(r.created_at).toLocaleDateString("en-US", { timeZone: "America/New_York", year: "numeric", month: "short", day: "numeric" }) : "Date unavailable"}</td></tr>}<tr>
                      <td><Link href={`/scanner/history/${r.id}`}>{(r.label || r.name || `Scan ${r.id}`).replace(/\bSP500\b/g, "S&P 500")}</Link></td>
                      <td className="cap">{etTime(r.created_at)}</td>
                      <td className="num mono">{r.row_count ?? "—"}</td>
                      <td className="num mono hide-narrow">{r.duration_s != null ? r.duration_s >= 60 ? `${Math.floor(r.duration_s / 60)}m ${Math.round(r.duration_s % 60)}s` : `${Math.round(r.duration_s)}s` : "—"}</td>
                    </tr></Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
    </div>
  );
}

export function SavedScanView({ id }: { id: number }) {
  const { can } = useSession();
  const allowed = can("can_scan_history");
  const premium = can("can_early_breakout");
  const run = useApi(allowed && id > 0 ? `run:${id}` : null, (signal) =>
    unwrap(api.GET("/v1/runs/{run_id}", { params: { path: { run_id: id } }, signal })));
  const r = run.data;
  return (
    <div className="stack">
      <Link href="/scanner/history" className="back">← Back to Scan history</Link>
      <Head title={r ? r.label || r.name || `Scan ${r.id}` : "Saved scan"} sub={r ? `Ran ${etTime(r.created_at)} · ${r.total} setup${r.total === 1 ? "" : "s"} ranked. Prices are from that scan.` : undefined} />
      {r && <p className="notice"><strong>Archived snapshot</strong> · {r.label || r.name || "Custom scan"} · Saved {etTime(r.created_at)}. Table prices and scores belong to this run. Opening a ticker shows its current research page.</p>}
      {!allowed ? LOCKED
        : !(id > 0) ? <Card><Empty title="That isn't a saved scan." /></Card>
        : run.error && !r ? (run.error.status === 404 ? <Card><Empty title="Saved scan not found.">It may have been removed, or it belongs to another account.</Empty></Card>
          : <ErrorState error={run.error} onRetry={run.reload} what="this scan" />)
        : !r ? <Skeleton rows={10} label="Loading the saved scan" />
        : r.setups.length === 0 ? <Card><Empty title="This scan had no ranked setups." /></Card>
        : <section className="card flush"><SetupTable rows={r.setups} premium={premium} /></section>}
      {r?.limited && (
        <div className="cap-note" role="note">
          <p>Your plan shows the top <b>{r.max_results}</b> of <b>{r.total}</b> setups in this scan.</p>
          <UpgradeButton plan="premium" />
        </div>
      )}
      {r && r.setups.length > 0 && <AIScanPanel runId={r.id} />}
    </div>
  );
}
