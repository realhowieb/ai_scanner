"use client";

// Pro: the Scanner's current results (every page the filters leave, up to the plan's cap)
// as a CSV file. Prices are the scan's, not live quotes, and the file says so.
import { ApiError, api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import type { paths } from "@/api/schema";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { downloadText, toCsv } from "@/lib/csv";
import { setupLabel } from "@/lib/format";

type Row = Schemas["ScanSetup"];
type Query = Omit<NonNullable<paths["/v1/scans/latest"]["get"]["parameters"]["query"]>, "limit" | "offset">;

const PAGE = 200;

/** Rounded for a spreadsheet; missing or non-finite stays blank (never 0). */
const round = (v: number | null | undefined, digits: number) => (v == null || !Number.isFinite(v) ? null : Number(v.toFixed(digits)));

export function scannerCsv(rows: Row[], scanAt: string | null | undefined, premium: boolean): string {
  const header = ["Rank", "Ticker", "HSF Score", "Setup", "Status", "Signals", "Scan price", "Change % at scan", "Gap %", "Relative volume", "Breakout score",
    ...(premium ? ["PreBreakout % (model estimate)"] : []), "Scanned at (UTC)"];
  return toCsv(header, rows.map((r, i) => [i + 1, r.ticker, r.score, setupLabel(r.primary_setup), r.status ?? "", r.signals.map(setupLabel).join("; "),
    round(r.last, r.last != null && Math.abs(r.last) < 1 ? 4 : 2), round(r.chg_pct, 2), round(r.gap_pct, 2), round(r.rvol, 2), round(r.breakout_score, 1),
    ...(premium ? [round(r.prob, 1)] : []), scanAt ?? ""]));
}

/** "hsf-scanner-2026-10-30-1549-ET.csv" from the scan time. */
export function csvName(scanAt: string | null | undefined): string {
  const d = scanAt ? new Date(scanAt) : null;
  if (!d || Number.isNaN(d.getTime())) return "hsf-scanner.csv";
  const part = (o: Intl.DateTimeFormatOptions) => d.toLocaleString("en-CA", { timeZone: "America/New_York", ...o });
  const day = part({ year: "numeric", month: "2-digit", day: "2-digit" });
  const time = part({ hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).replace(":", "");
  return `hsf-scanner-${day}-${time}-ET.csv`;
}

export function ScannerCsvButton({ query, total, scanAt, premium }: { query: Query; total: number; scanAt?: string | null; premium: boolean }) {
  const act = useAction();
  const run = () => void act.run(async () => {
    const rows: Row[] = [];
    let at = scanAt;
    for (let offset = 0; offset < total; offset += PAGE) {
      const page = await unwrap(api.GET("/v1/scans/latest", { params: { query: { ...query, limit: Math.min(PAGE, total - offset), offset } } }));
      // A new scan landed mid-download: start the file from that scan instead of mixing two.
      if (at && page.scan_at && page.scan_at !== at && offset > 0) throw new ApiError(0, "A new scan finished while downloading. Try again.", null, null, null);
      at = page.scan_at ?? at;
      rows.push(...page.setups);
      if (page.setups.length === 0) break;
    }
    downloadText(csvName(at), scannerCsv(rows, at, premium));
    return true;
  });
  return (
    <span className="csv-export">
      <button type="button" className="btn btn-sm" onClick={run} disabled={act.busy}
        aria-label={`Download ${total} scanner result${total === 1 ? "" : "s"} as CSV`}>{act.busy ? "Preparing…" : "Download CSV"}</button>
      <ErrorLine error={act.error} />
    </span>
  );
}
