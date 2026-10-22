"use client";

import { api, unwrap } from "@/api/client";
import { Card, Pill } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { ago, compact, etTime, price, shortDate } from "@/lib/format";

/** Recent headlines for the ticker (Alpaca News, Benzinga feed). Hidden when news is off. */
export function StockNews({ ticker }: { ticker: string }) {
  const res = useApi(`news:${ticker}`, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}/news", { params: { path: { ticker }, query: { limit: 6 } }, signal })));
  const d = res.data;
  if (!d || !d.available) return null;
  return (
    <Card title="News" id="news" aside="Benzinga via Alpaca">
      {d.items.length === 0 ? <p className="cap">No news for {ticker} in the last two weeks.</p> : (
        <ul className="news-list">
          {d.items.map((n) => (
            <li key={n.id || n.headline}>
              {n.url ? <a href={n.url} target="_blank" rel="noopener noreferrer nofollow">{n.headline}</a> : <span>{n.headline}</span>}
              <p className="cap">{ago(n.published_at) ?? etTime(n.published_at)}{n.source ? ` · ${n.source}` : ""}</p>
            </li>
          ))}
        </ul>
      )}
      <p className="cap">Headlines are third-party news, not HSF analysis.</p>
    </Card>
  );
}

const money = (v: number | null | undefined) => (v == null ? "—" : `$${compact(v)}`);

/** Insider buys and sales (Form 4) and company events (8-K) from SEC EDGAR, last 90 days. */
export function StockFilings({ ticker }: { ticker: string }) {
  const res = useApi(`filings:${ticker}`, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}/filings", { params: { path: { ticker } }, signal })));
  const d = res.data;
  if (!d || !d.available) return null;
  const s = d.insider_summary;
  return (
    <Card title="SEC filings" id="filings" aside={`Last ${s.days} days · EDGAR`}>
      <div className="stack-sm">
        <p className="body-sm">
          {s.buys || s.sales
            ? <>Insiders: <strong className="up">{s.buys} buy{s.buys === 1 ? "" : "s"}</strong> ({money(s.buy_value)}) · <strong className="down">{s.sales} sale{s.sales === 1 ? "" : "s"}</strong> ({money(s.sale_value)})</>
            : "No open-market insider buys or sales reported."}
        </p>
        {d.insider.length > 0 && (
          <ul className="filings-list">
            {d.insider.slice(0, 6).map((t, i) => (
              <li key={`${t.insider}-${t.date}-${i}`}>
                <Pill tone={t.type === "Buy" ? "up" : "down"}>{t.type}</Pill>{" "}
                <span>{t.insider}</span> <span className="cap">{t.role}</span>
                <p className="cap">
                  {shortDate(t.date)} · {t.shares != null ? `${compact(t.shares)} sh` : "—"}{t.price != null ? ` at ${price(t.price)}` : ""}{t.value != null ? ` · ${money(t.value)}` : ""}
                  {t.url && <> · <a href={t.url} target="_blank" rel="noopener noreferrer">filing</a></>}
                </p>
              </li>
            ))}
          </ul>
        )}
        {d.events.length > 0 && <>
          <h3 className="list-head">Company events (8-K)</h3>
          <ul className="filings-list">
            {d.events.slice(0, 5).map((e) => (
              <li key={e.url}>
                <a href={e.url} target="_blank" rel="noopener noreferrer">{e.labels.join(", ")}</a>
                <p className="cap">{shortDate(e.date)}</p>
              </li>
            ))}
          </ul>
        </>}
        <p className="cap">Option grants, gifts and exercises are left out. Filings can lag the trade by up to two business days.</p>
      </div>
    </Card>
  );
}
