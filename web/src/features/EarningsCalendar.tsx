"use client";

// Earnings calendar (Scanner > Earnings > Calendar): the next five trading days side by side,
// each day's biggest reporters first (by dollar volume) as logo tiles, plus your watchlist
// names. Pro (can_earnings), like the rest of the week ahead.
import Link from "@/components/Link";
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, ErrorState, Locked, ScoreBadge, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";

type Day = Schemas["EarningsWeekDay"];
type Name = Schemas["EarningsWeekName"];

const PER_DAY = 16;
const TIME: Record<string, { icon: string; label: string; cls: string }> = {
  bmo: { icon: "☀", label: "Before open", cls: "earn-bmo" },
  amc: { icon: "☾", label: "After close", cls: "earn-amc" },
};

function dayTitle(date: string): { weekday: string; rest: string } {
  const d = new Date(`${date}T12:00:00Z`);
  return {
    weekday: d.toLocaleDateString("en-US", { timeZone: "UTC", weekday: "long" }),
    rest: d.toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" }),
  };
}

function Logo({ ticker, kind }: { ticker: string; kind: Name["logo"] }) {
  const [failed, setFailed] = useState(false);
  // "none": the API found no usable logo (missing, or a photo), so skip the request.
  if (failed || kind === "none") return <span className="ecal-logo ecal-letter" aria-hidden="true">{ticker.slice(0, 1)}</span>;
  // Same-origin proxy (api/logo): the CSP allows only our own images. White logos sit on a dark chip.
  // eslint-disable-next-line @next/next/no-img-element
  return <img className={`ecal-logo${kind === "light" ? " ecal-logo-dark" : ""}`} src={`/api/logo/${encodeURIComponent(ticker)}`} alt="" width={32} height={32} loading="lazy" onError={() => setFailed(true)} />;
}

function Tile({ n }: { n: Name }) {
  const t = n.time ? TIME[n.time] : undefined;
  const label = `${n.ticker}, ${t ? t.label.toLowerCase() : "time not announced"}${n.score != null ? `, HSF Score ${n.score}` : ""}${n.on_watchlist ? ", on your watchlist" : ""}`;
  return (
    <li>
      <Link href={`/stocks/${encodeURIComponent(n.ticker)}`} className={`ecal-tile${n.on_watchlist ? " ecal-mine" : ""}`} aria-label={label}>
        <Logo ticker={n.ticker} kind={n.logo} />
        <span className="ecal-ticker">{n.on_watchlist && <span aria-hidden="true">★ </span>}{n.ticker}</span>
        <span className="ecal-meta" aria-hidden="true">
          {t ? <span className={`ecal-time ${t.cls}`} title={t.label}>{t.icon}</span> : <span className="ecal-time cap" title="Time not announced">TBA</span>}
          {n.score != null && <ScoreBadge score={n.score} />}
        </span>
      </Link>
    </li>
  );
}

function DayColumn({ day }: { day: Day }) {
  const { weekday, rest } = dayTitle(day.date);
  const more = day.total - day.shown;
  return (
    <section className="ecal-day" aria-label={`${weekday} ${rest}`}>
      <header className="ecal-head">
        <h3><span>{weekday}</span> <span className="cap">{rest}</span></h3>
        <span className="cap">{day.total} reporting</span>
      </header>
      {day.names.length > 0
        ? <ul className="ecal-grid">{day.names.map((n) => <Tile key={n.ticker} n={n} />)}</ul>
        : <p className="cap ecal-none">No companies we track report this day.</p>}
      {more > 0 && <p className="cap ecal-more">+ {more} more</p>}
    </section>
  );
}

export function EarningsCalendar({ pro }: { pro: boolean }) {
  const res = useApi(pro ? `earnings-week:${PER_DAY}` : null, (signal) =>
    unwrap(api.GET("/v1/earnings/week", { params: { query: { per_day: PER_DAY } }, signal })));
  if (!pro) {
    return (
      <Card><Locked title="The earnings calendar is part of Pro" plan="pro">
        See every big company reporting this week at a glance, day by day, with before-open and after-close timing.
      </Locked></Card>
    );
  }
  const { data, error, loading, reload } = res;
  if (error && !data) return <ErrorState error={error} onRetry={reload} what="the earnings calendar" />;
  if (loading && !data) return <Skeleton rows={6} label="Loading the earnings calendar" />;
  if (!data) return null;
  const total = data.days.reduce((s, d) => s + d.total, 0);
  return (
    <>
      <div className="ecal" role="region" aria-label="Earnings this week">
        {data.days.map((d) => <DayColumn key={d.date} day={d} />)}
      </div>
      <p className="cap">
        {total} companies report in the next five trading days. Each day shows its biggest names first, by trading volume, plus your
        watchlist (★). ☀ reports before the open, ☾ after the close. Scores follow your plan. Dates come from the earnings calendar and can move.{" "}
        <Link href="/scanner?signal=earnings&when=week&all=1">See every reporter in the table</Link>
      </p>
    </>
  );
}
