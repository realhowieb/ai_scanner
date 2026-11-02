"use client";

// Admin Marketing tab (admins only): where visitors, email-list sign-ups, accounts and sales
// come from, per source and per utm campaign; the email list's size and CSV export; and a
// link builder that adds utm tags to a post's link (for Buffer or any scheduler).
import { useMemo, useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { downloadText } from "@/lib/csv";
import { siteOrigin } from "@/lib/seo";

type Report = Schemas["AdminMarketing"];
const PERIODS = [[7, "7 days"], [30, "30 days"], [90, "90 days"], [365, "1 year"]] as const;
const fmt = (n: number) => n.toLocaleString("en-US");
const money = (cents: number) => `$${(cents / 100).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

/** utm_source values ui/acquisition.py credits to each site. */
export const CHANNELS = [["x", "X"], ["reddit", "Reddit"], ["stocktwits", "Stocktwits"], ["youtube", "YouTube"],
  ["threads", "Threads"], ["bluesky", "Bluesky"], ["tiktok", "TikTok"], ["discord", "Discord"],
  ["facebook", "Facebook / Instagram"], ["linkedin", "LinkedIn"],
  ["newsletter", "Newsletter"]] as const;
export const PAGES = [["/", "Home page"], ["/weekly", "Weekly recap"], ["/pricing", "Pricing and Founding 100"],
  ["/founders", "Founders Wall"], ["/demo", "Demo"]] as const;

/** The page link with utm tags; the campaign is lower-cased with spaces as dashes. */
export function taggedLink(origin: string, page: string, source: string, campaign: string): string {
  const url = new URL(page, origin);
  url.searchParams.set("utm_source", source);
  url.searchParams.set("utm_medium", source === "newsletter" ? "email" : "social");
  const c = campaign.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
  if (c) url.searchParams.set("utm_campaign", c);
  return url.toString();
}

function LinkBuilder() {
  const [page, setPage] = useState<string>("/weekly");
  const [source, setSource] = useState<string>("x");
  const [campaign, setCampaign] = useState("");
  const [copied, setCopied] = useState(false);
  const link = useMemo(() => taggedLink(siteOrigin({ NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL }), page, source, campaign),
    [page, source, campaign]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* select the text instead */
    }
  };
  return (
    <Card title="Campaign link builder" id="link-builder">
      <p className="body-sm">Use this link in a post so its visits and sign-ups show up under its source and campaign below.</p>
      <div className="inline-form">
        <label className="field"><span>Page</span>
          <select value={page} onChange={(e) => setPage(e.target.value)}>{PAGES.map(([p, l]) => <option key={p} value={p}>{l}</option>)}</select></label>
        <label className="field"><span>Where you&apos;ll post it</span>
          <select value={source} onChange={(e) => setSource(e.target.value)}>{CHANNELS.map(([s, l]) => <option key={s} value={s}>{l}</option>)}</select></label>
        <label className="field grow"><span>Campaign name</span>
          <input value={campaign} maxLength={60} placeholder="weekly-recap-w45" onChange={(e) => setCampaign(e.target.value)} /></label>
      </div>
      <div className="inline-form">
        <input className="grow mono" readOnly value={link} aria-label="Tagged link" onFocus={(e) => e.currentTarget.select()} />
        <button type="button" className="btn" onClick={() => void copy()}>{copied ? "Copied" : "Copy link"}</button>
      </div>
    </Card>
  );
}

function ListCard({ r }: { r: Report }) {
  const act = useAction();
  const download = () => void act.run(async () => {
    const res = await api.GET("/v1/admin/marketing/subscribers.csv", { params: { query: { status: "confirmed" } }, parseAs: "text" });
    if (res.error !== undefined || typeof res.data !== "string") throw new Error("Couldn't download the list.");
    downloadText("hsf-email-list-confirmed.csv", res.data);
    return true;
  });
  const l = r.list;
  return (
    <Card title="Weekly setups email list" id="email-list"
      aside={l.sync_configured ? <Pill tone="up">Syncing to Resend</Pill> : <Pill tone="warn">Not synced</Pill>}>
      <div className="tiles">
        {([["Confirmed", l.confirmed], ["Waiting to confirm", l.pending], ["Unsubscribed", l.unsubscribed], [`New in ${r.days} days`, l.new_in_period]] as const).map(([k, v]) => (
          <div key={k} className="tile"><span className="cap">{k}</span><p className="tile-value small">{fmt(v)}</p></div>
        ))}
      </div>
      <p className="cap">{l.sync_configured
        ? "Each address is copied to your Resend segment when it's confirmed, ready for a Broadcast."
        : "Set HSF_RESEND_SEGMENT_ID (and a full-access RESEND_API_KEY) on the API to copy confirmed addresses to Resend automatically, or download the CSV and import it."}</p>
      <div className="row-actions"><button type="button" className="btn" disabled={act.busy} onClick={download}>Download confirmed (CSV)</button></div>
      <ErrorLine error={act.error} />
    </Card>
  );
}

function Sources({ r }: { r: Report }) {
  const t = r.totals;
  return (
    <Card title="Where people come from" id="sources" aside={`Last ${r.days} days`}>
      {r.by_source.length === 0 ? <p className="cap">Nothing recorded in this period yet.</p> : (
        <div className="table-wrap"><table className="table">
          <thead><tr><th scope="col">Source</th><th scope="col" className="num">Visits</th><th scope="col" className="num">Sign-up clicks</th>
            <th scope="col" className="num">Email list</th><th scope="col" className="num">Accounts</th><th scope="col" className="num">Paid</th>
            <th scope="col" className="num">Founders</th></tr></thead>
          <tbody>{r.by_source.map((s) => (
            <tr key={s.source}><th scope="row">{s.source}</th><td className="num mono">{fmt(s.visits)}</td><td className="num mono">{fmt(s.clicks)}</td>
              <td className="num mono">{fmt(s.list_signups)}{s.list_signups > 0 && <span className="cap"> ({fmt(s.list_confirmed)} confirmed)</span>}</td>
              <td className="num mono">{fmt(s.accounts)}</td><td className="num mono">{fmt(s.paid)}</td><td className="num mono">{fmt(s.founders)}</td></tr>
          ))}</tbody>
          <tfoot><tr><th scope="row">Total</th><td className="num mono">{fmt(t.visits)}</td><td className="num mono">{fmt(t.clicks)}</td>
            <td className="num mono">{fmt(t.list_signups)}</td><td className="num mono">{fmt(t.accounts)}</td><td className="num mono">{fmt(t.paid)}</td>
            <td className="num mono">{fmt(t.founders)}</td></tr></tfoot>
        </table></div>
      )}
      {r.founder_revenue_cents > 0 && <p className="body-sm">Founding 100 sales in this period: <strong>{money(r.founder_revenue_cents)}</strong>.</p>}
      <p className="cap">Visits count browser sessions on public pages, not people. Paid plans and founder seats are credited to the source the buyer signed up from, whenever that was. Admin and app sign-ups without a first visit show as Direct.</p>
      {r.unavailable.length > 0 && <p className="cap">Couldn&apos;t load {r.unavailable.join(", ")}; those count as zero.</p>}
    </Card>
  );
}

function Campaigns({ r }: { r: Report }) {
  return (
    <Card title="Campaigns" id="campaigns" aside="Links with a utm_campaign tag">
      {r.campaigns.length === 0 ? <p className="cap">No tagged links used yet. Make one with the link builder.</p> : (
        <div className="table-wrap"><table className="table">
          <thead><tr><th scope="col">Campaign</th><th scope="col">Source</th><th scope="col" className="num">Visits</th>
            <th scope="col" className="num">Sign-up clicks</th><th scope="col" className="num">Email list</th><th scope="col" className="num">Accounts</th></tr></thead>
          <tbody>{r.campaigns.map((c) => (
            <tr key={`${c.utm_source}/${c.campaign}`}><th scope="row" className="mono">{c.campaign}</th><td>{c.utm_source}</td>
              <td className="num mono">{fmt(c.visits)}</td><td className="num mono">{fmt(c.clicks)}</td>
              <td className="num mono">{fmt(c.list_signups)}</td><td className="num mono">{fmt(c.accounts)}</td></tr>
          ))}</tbody>
        </table></div>
      )}
    </Card>
  );
}

export function MarketingTab() {
  const [days, setDays] = useState(30);
  const s = useApi<Report>(`admin-marketing:${days}`, (signal) => unwrap(api.GET("/v1/admin/marketing", { params: { query: { days } }, signal })));
  return (
    <div className="stack">
      <div className="inline-form">
        <label className="field"><span>Period</span>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>{PERIODS.map(([d, l]) => <option key={d} value={d}>{l}</option>)}</select>
        </label>
      </div>
      {s.error && !s.data ? <ErrorState error={s.error} onRetry={s.reload} what="the marketing report" />
        : !s.data ? <Skeleton rows={5} label="Loading the marketing report" />
        : <><Sources r={s.data} /><Campaigns r={s.data} /><ListCard r={s.data} /></>}
      <LinkBuilder />
    </div>
  );
}
