"use client";

import Link from "next/link";

import type { Schemas } from "@/api/client";
import { Card, Disclaimer, Empty, Freshness, Locked, Pill, RANKED_MIN, ScoreBadge, ScoreBar, TickerLink } from "@/components/ui";
import { etDate, etTime, freshness, greeting, pct, price, setupLabel } from "@/lib/format";

import { SnapshotTiles, StatusStrip } from "./MarketSnapshot";

type Today = Schemas["Today"];
type SessionCard = Schemas["SessionCard"];
type Setup = Schemas["Setup"];
type Mine = Schemas["TodayPersonal"];

const PHASES: Record<string, { label: string; note: string; tone: string }> = {
  premarket: { label: "Pre-market", note: "Opens 9:30 AM ET", tone: "amber" },
  open: { label: "Market open", note: "Closes 4:00 PM ET", tone: "green" },
  afterhours: { label: "After hours", note: "Until 8:00 PM ET", tone: "amber" },
  closed: { label: "Market closed", note: "Next session 9:30 AM ET", tone: "grey" },
};

const CHIP_LIMIT = 12;

/** Recap names arrive as "VST (53)"; each chip opens that stock. */
function TickerChips({ items }: { items: string[] }) {
  const shown = items.slice(0, CHIP_LIMIT);
  return (
    <div className="chips">
      {shown.map((e) => {
        const m = /^(\S+)\s*\((\d+)\)$/.exec(e.trim());
        const ticker = m ? m[1]! : e.trim();
        return (
          <Link key={e} href={`/stocks/${encodeURIComponent(ticker)}`} className="pill" aria-label={m ? `${ticker}, HSF ${m[2]}` : ticker}>
            {ticker}{m && <span className="chip-score"> {m[2]}</span>}
          </Link>
        );
      })}
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

function NewSince({ data }: { data: Schemas["NewSince"] }) {
  return (
    <Card title="New since your last visit" id="new" aside={data.baseline_scan_at ? `Compared with the scan from ${etTime(data.baseline_scan_at)}` : undefined}>
      {data.tickers.length === 0 ? (
        <p className="cap">Nothing new since you last looked, or this is your first visit on this browser.</p>
      ) : (
        <>
          <p className="body-sm"><b>{data.total}</b> name{data.total === 1 ? "" : "s"} in the latest scan weren&apos;t in the one you saw last, strongest first.</p>
          <TickerChips items={data.tickers.map((t) => (t.score !== null && t.score !== undefined ? `${t.ticker} (${t.score})` : t.ticker))} />
        </>
      )}
    </Card>
  );
}

const MISSING_LIMIT = 20;

function YourWatchlist({ data }: { data: Schemas["WatchlistToday"] }) {
  const s = data.summary;
  return (
    <Card title="Your watchlist" id="wl" aside={data.watchlist_id ? <Link href="/watchlists">{data.name}</Link> : undefined}>
      {!data.watchlist_id || (data.in_scan.length === 0 && data.missing.length === 0) ? (
        <Empty title="Your watchlist is empty.">Add names from the Scanner or a stock page.</Empty>
      ) : (
        <>
          {s && <p className="cap">{s.tracked} watched · {s.needs_attention} need attention · {s.strengthening} strengthening · {s.fading} fading</p>}
          {data.in_scan.length > 0 ? (
            <ul className="rows">
              {data.in_scan.map((r) => (
                <li key={r.ticker} className="row">
                  <TickerLink ticker={r.ticker} />
                  <span className="cap grow">In the latest scan</span>
                  <span className="cap">HSF</span><ScoreBadge score={r.score} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="cap">None of your watched names are in the latest scan.</p>
          )}
          {data.missing.length > 0 && (
            <p className="cap">Not in the latest scan: {data.missing.slice(0, MISSING_LIMIT).join(", ")}
              {data.missing.length > MISSING_LIMIT ? ` and ${data.missing.length - MISSING_LIMIT} more` : ""}</p>
          )}
        </>
      )}
    </Card>
  );
}

function SetupRow({ s, muted = false }: { s: Setup; muted?: boolean }) {
  return (
    <tr className={muted ? "row-muted" : undefined}>
      <td><TickerLink ticker={s.ticker} /></td>
      <td className="hide-narrow"><Pill>{setupLabel(s.primary_setup)}</Pill></td>
      <td><ScoreBar score={s.score} /></td>
      <td className="num mono hide-narrow">{price(s.last)}</td>
      <td className={`num mono ${s.chg_pct && s.chg_pct > 0 ? "up" : s.chg_pct && s.chg_pct < 0 ? "down" : ""}`}>{pct(s.chg_pct)}</td>
    </tr>
  );
}

/** `mine` is the signed-in sections (GET /v1/today/me); they load separately and never hold up the page. */
export function TodayView({ data, mine, mineFailed = false }: { data: Today; mine?: Mine | null; mineFailed?: boolean }) {
  const failed = new Set(data.errors.map((e) => e.section));
  const mineErr = new Set(mine?.errors.map((e) => e.section) ?? (mineFailed ? ["new_since", "watchlist"] : []));
  const phase = PHASES[data.market.phase] ?? PHASES.closed!;
  const top = data.top_setups;
  const strongMin = top?.threshold ?? 75;
  const also = top?.also_ranked ?? [];
  const alsoMin = top?.ranked_floor ?? RANKED_MIN;
  const shownOnTop = new Set([...(top?.setups ?? []), ...also].map((s) => s.ticker));
  // Standouts are the recapped session's top names; skip them when Top setups already shows the same list.
  const standouts = (data.recap?.standouts ?? []).filter((o) => !shownOnTop.has(o.ticker));
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

      <StatusStrip market={data.market} snapshot={data.snapshot} />

      {stale && (
        <p className="banner" role="status">
          Market data is delayed: the latest market scan is from {etTime(scanAt)}
          {data.market.expected_scan_at ? ` and the ${etTime(data.market.expected_scan_at)} scan hasn't arrived yet` : ""}.
          Scores and prices below are from that scan and may be out of date.
        </p>
      )}

      {failed.has("snapshot") ? <SectionFailed name="Market snapshot" /> : data.snapshot && (
        <Card title="Market snapshot" id="snap" aside="Top gainer and most active are from the latest full-market scan">
          <SnapshotTiles snapshot={data.snapshot} />
        </Card>
      )}

      <div className="split today-split">
        <div className="col-main">
          {failed.has("before_open") ? <SectionFailed name="Before the open" /> : data.before_open && (
            <Movers card={data.before_open} id="bto" title="Before the open" caption="Pre-market scan vs the previous close. Pre-market prices keep moving." />
          )}

          <Card title="Top setups" id="top" aside={top?.scan_at ? <Freshness at={top.scan_at} label="Full-market scan" stale={data.market.stale} /> : "Ranked by HSF Score"}>
            {failed.has("top_setups") || !top ? (
              <SectionFailed name="Top setups" />
            ) : top.state === "empty_scan" ? (
              <Empty title="No scan results yet.">The next scheduled scan will fill this in.</Empty>
            ) : top.state === "no_qualifying" && also.length === 0 ? (
              <Empty title={`No setup reached HSF ${strongMin} in the latest scan.`}>That happens on quiet days. The Scanner still lists every ranked name.</Empty>
            ) : (
              <>
                {top.setups.length === 0 && <p className="cap">No setup reached HSF {strongMin} in the latest scan. These are the highest ranked names.</p>}
                <div className="table-wrap">
                  <table className="table">
                    <thead>
                      <tr><th scope="col">Ticker</th><th scope="col" className="hide-narrow">Setup</th><th scope="col" className="w40">HSF Score</th><th scope="col" className="num hide-narrow">Last</th><th scope="col" className="num">Chg</th></tr>
                    </thead>
                    {top.setups.length > 0 && (
                      <tbody>
                        {top.setups.map((s) => <SetupRow key={s.ticker} s={s} />)}
                      </tbody>
                    )}
                    {also.length > 0 && (
                      <tbody className="also">
                        {top.setups.length > 0 && <tr className="sub-head"><th scope="rowgroup" colSpan={5}>Also ranked</th></tr>}
                        {also.map((s) => <SetupRow key={s.ticker} s={s} muted />)}
                      </tbody>
                    )}
                  </table>
                </div>
                <p className="cap">Strong setups score HSF {strongMin}+.{also.length > 0 ? ` Also ranked: HSF ${alsoMin} to ${strongMin - 1}.` : ""}</p>
              </>
            )}
            <div className="card-foot">
              <Disclaimer />
              <Link href="/scanner">Open the Scanner →</Link>
            </div>
          </Card>

          {mineErr.has("new_since") ? <SectionFailed name="New since your last visit" /> : mine?.new_since && <NewSince data={mine.new_since} />}
          {mineErr.has("watchlist") ? <SectionFailed name="Your watchlist" /> : mine?.watchlist && <YourWatchlist data={mine.watchlist} />}
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
              {standouts.length > 0 && (
                <div className="chips-block"><p className="cap">Strongest in the last scan</p>
                  <TickerChips items={standouts.map((o) => `${o.ticker} (${o.score})`)} /></div>
              )}
              {data.recap.entered.length > 0 && (
                <div className="chips-block"><p className="cap">Entered the ranked list (HSF {RANKED_MIN}+), with their latest score</p>
                  <TickerChips items={data.recap.entered} /></div>
              )}
              {data.recap.left.length > 0 && (
                <div className="chips-block"><p className="cap">Left the ranked list, with their score from the day&apos;s first scan</p>
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
