"use client";

// One-tap starters for empty pages: a first watchlist from today's top setups, and
// ready-made alerts, so a new account has something working without filling in forms.
import Link from "@/components/Link";
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import { alertRules, watchlists } from "@/api/userData";
import type { AlertRuleCreate } from "@/api/userData";
import { Card, ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { useSession } from "@/session/SessionProvider";

const STARTER_COUNT = 5;
const STARTER_NAME = "Today's top setups";

/** Adds today's top ranked tickers to `listId`, or to a new default list when there is none. */
export function StarterWatchlistButton({ tickers, listId, onDone }: { tickers?: string[]; listId?: number | null; onDone: (id: number) => void }) {
  const latest = useApi(tickers ? null : "starter-top", (signal) =>
    unwrap(api.GET("/v1/scans/latest", { params: { query: { limit: STARTER_COUNT } }, signal })));
  const act = useAction();
  const picks = (tickers ?? latest.data?.setups.map((s) => s.ticker) ?? []).slice(0, STARTER_COUNT);
  if (!picks.length) return null;
  const start = () => void act.run(async () => {
    const id = listId ?? (await watchlists.create(STARTER_NAME, true)).id;
    await watchlists.addTickers(id, picks);
    onDone(id);
    return true;
  });
  return (
    <div className="stack-sm">
      <div className="row-actions">
        <button type="button" className="btn btn-primary" onClick={start} disabled={act.busy}>
          {act.busy ? "Adding…" : `Start with today's top ${picks.length}`}
        </button>
      </div>
      <p className="cap">Adds {picks.join(", ")} from the latest scan. Remove any you don&apos;t want.</p>
      <ErrorLine error={act.error} />
    </div>
  );
}

type Starter = { key: string; label: string; detail: string; body: AlertRuleCreate };

/** Ready-made smart alerts for an account with none. Watchlist ones need a list with tickers. */
export function StarterAlerts({ onDone }: { onDone: (label: string) => void }) {
  const { can } = useSession();
  const lists = useApi("watchlists", (signal) => watchlists.list(signal));
  const rules = useApi("alert-rules", (signal) => alertRules.list(signal));
  const act = useAction();
  const [made, setMade] = useState<string[]>([]);
  if (!lists.data || !rules.data) return null;
  // Only for accounts with no smart alerts yet (keep showing while this card adds them).
  if (made.length === 0 && (rules.data.rules ?? []).length > 0) return null;
  const list = lists.data.find((w) => w.is_default) ?? lists.data[0];
  const hasTickers = !!list && (list.symbol_count ?? 0) > 0;
  const starters: Starter[] = [];
  if (hasTickers) {
    starters.push({
      key: "wl-75", label: `A stock on ${list.name} scores 75 or more`, detail: "Checked after each market scan; fires once when a score crosses 75.",
      body: { rule_type: "HSF_SCORE_CROSS_ABOVE", watchlist_id: list.id, threshold: 75, enabled: true, delivery_channels: ["in_app"] },
    }, {
      key: "wl-rvol", label: `A stock on ${list.name} trades 2x its usual volume`, detail: "Unusual volume often comes before a move.",
      body: { rule_type: "RVOL_ABOVE", watchlist_id: list.id, threshold: 2, enabled: true, delivery_channels: ["in_app"] },
    });
  }
  if (can("can_scan_alerts")) {
    starters.push({
      key: "scan-breakout", label: "A new Breakout setup scores 70 or more", detail: "Any stock in the full-market scan, once per scan with the new names.",
      body: { rule_type: "SCAN_MATCH", threshold: 70, value: JSON.stringify({ setup: "breakout" }), enabled: true, delivery_channels: ["in_app"] },
    });
  }
  const add = (s: Starter) => void act.run(async () => {
    await alertRules.create(s.body);
    setMade((m) => [...m, s.key]);
    onDone(s.label);
    return true;
  });
  return (
    <Card title="Quick start" id="starters">
      <p className="cap">Ready-made alerts. One tap each; change or delete them any time under Rules.</p>
      {starters.length ? (
        <ul className="alert-list">
          {starters.map((s) => (
            <li key={s.key} className="alert-row">
              <div className="grow stack-xs"><span className="strong">{s.label}</span><span className="cap">{s.detail}</span></div>
              {made.includes(s.key) ? <span className="cap" role="status">Added</span>
                : <button type="button" className="btn btn-sm" onClick={() => add(s)} disabled={act.busy}>Add</button>}
            </li>
          ))}
        </ul>
      ) : null}
      {!hasTickers && (
        <p className="cap">Watchlist alerts need a list with tickers. <Link href="/watchlists">Start a watchlist</Link> first, then come back.</p>
      )}
      <ErrorLine error={act.error} />
    </Card>
  );
}
