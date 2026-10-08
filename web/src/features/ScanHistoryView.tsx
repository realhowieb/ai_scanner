"use client";

// Scan history (Pro): your own saved scans (/v1/runs) and one saved scan's ranked rows
// (/v1/runs/{id}), shown with the same table as the Scanner.
import Link from "next/link";

import { api, unwrap } from "@/api/client";
import { Card, Disclaimer, Empty, ErrorState, Locked, Skeleton, UpgradeButton } from "@/components/ui";
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
  return (
    <div className="stack">
      <Head title="Scan history" sub="Your saved custom scans, newest first." />
      {!allowed ? LOCKED
        : runs.error && !runs.data ? (runs.error.status === 403 ? LOCKED : <ErrorState error={runs.error} onRetry={runs.reload} what="your scan history" />)
        : !runs.data ? <Skeleton rows={6} label="Loading scan history" />
        : runs.data.length === 0 ? <Card><Empty title="No saved scans yet."><Link href="/scanner/custom">Run a custom scan</Link> and it will be saved here.</Empty></Card>
        : (
          <section className="card flush">
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Scan</th><th scope="col">Run</th><th scope="col" className="num">Rows</th><th scope="col" className="num hide-narrow">Took</th></tr></thead>
                <tbody>
                  {runs.data.map((r) => (
                    <tr key={r.id}>
                      <td><Link href={`/scanner/history/${r.id}`}>{r.label || r.name || `Scan ${r.id}`}</Link></td>
                      <td className="cap">{etTime(r.created_at)}</td>
                      <td className="num mono">{r.row_count ?? "—"}</td>
                      <td className="num mono hide-narrow">{r.duration_s != null ? `${Math.round(r.duration_s)}s` : "—"}</td>
                    </tr>
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
      <Disclaimer />
    </div>
  );
}
