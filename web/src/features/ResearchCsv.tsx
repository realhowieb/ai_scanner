"use client";

// Premium: CSV downloads of your own research (watchlists, journal, Track record, saved scans).
// The data is already on the page; the file is built in the browser. Other plans see a
// small link to the plans instead of the button.
import Link from "@/components/Link";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { downloadText, toCsv } from "@/lib/csv";
import { useSession } from "@/session/SessionProvider";

/** Rounded for a spreadsheet; missing or non-finite stays blank (never 0). */
export const round = (v: number | null | undefined, digits = 2) => (v == null || !Number.isFinite(v) ? null : Number(v.toFixed(digits)));

/** Today's date in New York, for file names: "2026-11-02". */
export function etDay(d = new Date()): string {
  return d.toLocaleDateString("en-CA", { timeZone: "America/New_York" });
}

/** "hsf-watchlist-tech-leaders-2026-11-02.csv" */
export function csvFile(kind: string, name = "", day = etDay()): string {
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
  return `hsf-${kind}${slug ? `-${slug}` : ""}-${day}.csv`;
}

export type CsvTable = { header: string[]; rows: unknown[][] };

export function ResearchCsvButton({ what, filename, build }: { what: string; filename: string; build: () => CsvTable | Promise<CsvTable> }) {
  const { can } = useSession();
  const act = useAction();
  if (!can("can_export_research")) {
    return <Link href="/pricing" className="btn btn-sm export-csv" aria-label={`CSV export of ${what} is part of Premium`}>CSV · Premium</Link>;
  }
  const run = () => void act.run(async () => {
    const t = await build();
    downloadText(filename, toCsv(t.header, t.rows));
    return true;
  });
  return (
    <span className="export-csv">
      <button type="button" className="btn btn-sm" onClick={run} disabled={act.busy} aria-label={`Download ${what} as CSV`}>
        {act.busy ? "Preparing…" : "Download CSV"}
      </button>
      <ErrorLine error={act.error} />
    </span>
  );
}
