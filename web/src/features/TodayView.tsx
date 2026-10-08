"use client";

import Link from "next/link";

import type { Schemas } from "@/api/client";
import { Card, Disclaimer, Empty, Freshness, Locked, Pill, ScoreBadge, ScoreBar, TickerLink } from "@/components/ui";
import { etDate, etTime, freshness, greeting, pct, price, setupLabel } from "@/lib/format";

type Today = Schemas["Today"];
type SessionCard = Schemas["SessionCard"];

const PHASES: Record<string, { label: string; note: string; tone: string }> = {
  premarket: { label: "Pre-market", note: "Opens 9:30 AM ET", tone: "amber" },
  open: { label: "Market open", note: "Closes 4:00 PM ET", tone: "green" },
  afterhours: { label: "After hours", note: "Until 8:00 PM ET", tone: "amber" },
  closed: { label: "Market closed", note: "Next session 9:30 AM ET", tone: "grey" },
};

const CHIP_LIMIT = 12;

function TickerChips({ items }: { items: string[] }) {
  const shown = items.slice(0, CHIP_LIMIT);
  return (
    <div className="chips">
      {shown.map((e) => <Pill key={e}>{e}</Pill>)}
      {items.length > shown.length && <span className="cap">+{items.length - shown.length} more</span>}
    </div>
  );
}

function SectionFailed({ name }: { name: string }) {
  return <p className="section-error" role="alert">{name} couldn&apos;t load right now. The rest of the page is current.</p>;
}

function Movers({ card, title, caption, id }: { card: SessionCard; title: string; caption: string; id: string }) {
  return (
    <Card title={title} id={id} aside={card.scan_at ? <Freshness at={card.scan_at} label="Scan" staleCheck={false} /> : undefined}>
      {card.locked ? (
        <Locked title={`${title} movers are part of Pro`} plan="pro">See which names are moving outside regular hours, with their HSF Score.</Locked>
      ) : card.movers.length === 0 ? (
        <Empty title="No movers in this session's scan yet." />
      ) : (
        <ul className="rows">
          {card.movers.map((m) => (
            <li key={m.ticker} className="row">
              <TickerLink ticker={m.ticker} />
              <span className={`mono strong move ${m.pct && m.pct > 0 ? "up" : m.pct && m.pct < 0 ? "down" : ""}`}>{pct(m.pct)}</span>
              <span className="mono cap grow">{price(m.last)}</span>
              {m.score !== null && m.score !== undefined && <><span className="cap">HSF</span><ScoreBadge score={m.score} /></>}
            </li>
          ))}
        </ul>
      )}
      <p className="cap">{caption}</p>
    </Card>
  );
}

export function TodayView({ data }: { data: Today }) {
  const failed = new Set(data.errors.map((e) => e.section));
  const phase = PHASES[data.market.phase] ?? PHASES.closed!;
  const top = data.top_setups;
  // The API knows the scan schedule (a missed slot, not overnight or weekend gaps);
  // the age rule is only a fallback for an API that predates market.stale.
  const scanAt = data.market.latest_scan_at ?? top?.scan_at;
  const stale = data.market.stale ?? (top?.scan_at ? freshness(top.scan_at).stale : false);

  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <p className="cap">{etDate(data.as_of)}</p>
          <h1 className="h1">{greeting(data.as_of)}</h1>
        </div>
        <div className={`phase phase-${phase.tone}`}>
          <span className="dot" aria-hidden="true" />
          <span className="strong">{phase.label}</span>
          <span className="cap">{phase.note}</span>
        </div>
      </section>

      {stale && (
        <p className="banner" role="status">
          Market data is delayed: the latest market scan is from {etTime(scanAt)}
          {data.market.expected_scan_at ? ` and the ${etTime(data.market.expected_scan_at)} scan hasn't arrived yet` : ""}.
          Scores and prices below are from that scan and may be out of date.
        </p>
      )}

      <div className="split">
        <div className="col-main">
          {failed.has("before_open") ? <SectionFailed name="Before the open" /> : data.before_open && (
            <Movers card={data.before_open} id="bto" title="Before the open" caption="Pre-market scan vs the previous close. Pre-market prices keep moving." />
          )}

          <Card title="Top setups" id="top" aside={top?.scan_at ? <Freshness at={top.scan_at} label="Full-market scan" stale={data.market.stale} /> : "Ranked by HSF Score"}>
            {failed.has("top_setups") || !top ? (
              <SectionFailed name="Top setups" />
            ) : top.state === "empty_scan" ? (
              <Empty title="No scan results yet.">The next scheduled scan will fill this in.</Empty>
            ) : top.state === "no_qualifying" ? (
              <Empty title={`No setup reached HSF ${top.threshold ?? 40} in the latest scan.`}>That happens on quiet days. The Scanner still lists every ranked name.</Empty>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th scope="col">Ticker</th><th scope="col" className="hide-narrow">Setup</th><th scope="col" className="w40">HSF Score</th><th scope="col" className="num hide-narrow">Last</th><th scope="col" className="num">Chg</th></tr>
                  </thead>
                  <tbody>
                    {top.setups.map((s) => (
                      <tr key={s.ticker}>
                        <td><TickerLink ticker={s.ticker} /></td>
                        <td className="hide-narrow"><Pill>{setupLabel(s.primary_setup)}</Pill></td>
                        <td><ScoreBar score={s.score} /></td>
                        <td className="num mono hide-narrow">{price(s.last)}</td>
                        <td className={`num mono ${s.chg_pct && s.chg_pct > 0 ? "up" : s.chg_pct && s.chg_pct < 0 ? "down" : ""}`}>{pct(s.chg_pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="card-foot">
              <Disclaimer />
              <Link href="/scanner">Open the Scanner →</Link>
            </div>
          </Card>
        </div>

        <aside className="col-side">
          {failed.has("after_close") ? <SectionFailed name="After the close" /> : data.after_close && (
            <Movers card={data.after_close} id="atc" title="After the close" caption="After-hours scan vs today's close. After-hours prices keep moving." />
          )}

          {failed.has("recap") ? <SectionFailed name="Last session recap" /> : data.recap && (
            <Card title="Last session recap" id="rc" aside={data.recap.title}>
              <p className="body-sm">
                <b>{data.recap.scans}</b> full-market scan{data.recap.scans === 1 ? "" : "s"} ran
                {data.recap.premarket_scans || data.recap.postmarket_scans
                  ? ` (plus ${data.recap.premarket_scans} pre-market and ${data.recap.postmarket_scans} after-hours)` : ""}.
              </p>
              {data.recap.entered.length > 0 && (
                <div className="chips-block"><p className="cap">Entered the ranked list (HSF 40+)</p>
                  <TickerChips items={data.recap.entered} /></div>
              )}
              {data.recap.left.length > 0 && (
                <div className="chips-block"><p className="cap">Left the ranked list</p>
                  <TickerChips items={data.recap.left} /></div>
              )}
              {data.recap.entered.length === 0 && data.recap.left.length === 0 && <p className="cap">No names entered or left the ranked list.</p>}
            </Card>
          )}
        </aside>
      </div>
    </div>
  );
}
