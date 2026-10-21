// Watchlist and alert calls, through the typed client.
import { captureUsage } from "@/lib/usageAnalytics";

import { api, unwrap } from "./client";
import type { Schemas } from "./client";

export type Watchlist = Schemas["Watchlist"];
export type WatchlistDetail = Schemas["WatchlistDetail"];
export type Alert = Schemas["Alert"];
export type AlertType = Schemas["AlertType"];
export type AlertCreate = Omit<Schemas["AlertCreate"], "watchlist_only"> & { watchlist_only?: boolean };
export type WatchlistIntelligence = Schemas["WatchlistIntelligence"];
export type WatchlistIntelItem = Schemas["WatchlistIntelItem"];
export type WatchlistChanges = Schemas["WatchlistChanges"];
export type AlertRule = Schemas["AlertRule"];
export type AlertRuleType = Schemas["AlertRuleType"];
export type AlertRuleCreate = Schemas["AlertRuleCreate"];
export type AlertRuleUpdate = Schemas["AlertRuleUpdate"];
export type WatchlistPerformance = Schemas["WatchlistPerformance"];
export type ScoreMovers = Schemas["ScoreMovers"];
export type JournalInsights = Schemas["JournalInsights"];
export type Referrals = Schemas["Referrals"];

const wid = (id: number) => ({ params: { path: { watchlist_id: id } } });

export const watchlists = {
  list: (signal?: AbortSignal) => unwrap(api.GET("/v1/watchlists", { signal })),
  get: (id: number, signal?: AbortSignal) => unwrap(api.GET("/v1/watchlists/{watchlist_id}", { ...wid(id), signal })),
  create: async (name: string, makeDefault = false) => {
    const result = await unwrap(api.POST("/v1/watchlists", { body: { name, make_default: makeDefault } }));
    captureUsage("watchlist_created");
    return result;
  },
  rename: (id: number, name: string) =>
    unwrap(api.PATCH("/v1/watchlists/{watchlist_id}", { ...wid(id), body: { name, make_default: false } })),
  makeDefault: (id: number) =>
    unwrap(api.PATCH("/v1/watchlists/{watchlist_id}", { ...wid(id), body: { make_default: true } as Schemas["WatchlistUpdate"] })),
  remove: async (id: number) => {
    const result = await unwrap(api.DELETE("/v1/watchlists/{watchlist_id}", wid(id)));
    captureUsage("watchlist_deleted");
    return result;
  },
  addTickers: async (id: number, tickers: string[]) => {
    const result = await unwrap(api.POST("/v1/watchlists/{watchlist_id}/tickers", { ...wid(id), body: { tickers } }));
    if (result.added.length > 0) captureUsage("watchlist_stock_added");
    return result;
  },
  removeTicker: async (id: number, ticker: string) => {
    const result = await unwrap(api.DELETE("/v1/watchlists/{watchlist_id}/tickers/{ticker}", { params: { path: { watchlist_id: id, ticker } } }));
    captureUsage("watchlist_stock_removed");
    return result;
  },
  intelligence: (id: number, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/watchlists/{watchlist_id}/intelligence", { ...wid(id), signal })),
  changes: (id: number, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/watchlists/{watchlist_id}/changes", { ...wid(id), signal })),
  performance: (id: number, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/watchlists/{watchlist_id}/performance", { ...wid(id), signal })),
  setNote: (id: number, ticker: string, note: string) =>
    unwrap(api.PATCH("/v1/watchlists/{watchlist_id}/tickers/{ticker}", {
      params: { path: { watchlist_id: id, ticker } }, body: { note: note.trim() ? note : null },
    })),
};

export const alerts = {
  list: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts", { signal })),
  types: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts/types", { signal })),
  events: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts/events", { params: { query: { limit: 20 } }, signal })),
  create: async (body: AlertCreate) => {
    const result = await unwrap(api.POST("/v1/alerts", { body: body as Schemas["AlertCreate"] }));
    captureUsage("alert_created", { kind: body.type === "price" ? "price" : "technical" });
    return result;
  },
  setEnabled: async (id: number, enabled: boolean) => {
    const result = await unwrap(api.PATCH("/v1/alerts/{alert_id}", { params: { path: { alert_id: id } }, body: { enabled } }));
    captureUsage("alert_enabled_changed", { enabled });
    return result;
  },
  remove: async (id: number) => {
    const result = await unwrap(api.DELETE("/v1/alerts/{alert_id}", { params: { path: { alert_id: id } } }));
    captureUsage("alert_deleted");
    return result;
  },
};

// Alert rules: conditions on HSF intelligence, evaluated by the server (never here).
export const alertRules = {
  list: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts/rules", { signal })),
  types: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts/rules/types", { signal })),
  create: async (body: AlertRuleCreate) => {
    const result = await unwrap(api.POST("/v1/alerts/rules", { body }));
    captureUsage("alert_created", { kind: body.rule_type === "SCAN_MATCH" ? "scan" : "hsf" });
    return result;
  },
  update: async (id: number, body: AlertRuleUpdate) => {
    const result = await unwrap(api.PATCH("/v1/alerts/rules/{rule_id}", { params: { path: { rule_id: id } }, body }));
    if (typeof body.enabled === "boolean") captureUsage("alert_enabled_changed", { enabled: body.enabled });
    return result;
  },
  setEnabled: async (id: number, enabled: boolean) => {
    const result = await unwrap(api.PATCH("/v1/alerts/rules/{rule_id}", { params: { path: { rule_id: id } }, body: { enabled } }));
    captureUsage("alert_enabled_changed", { enabled });
    return result;
  },
  remove: async (id: number) => {
    const result = await unwrap(api.DELETE("/v1/alerts/rules/{rule_id}", { params: { path: { rule_id: id } } }));
    captureUsage("alert_deleted");
    return result;
  },
};

export const insights = {
  movers: (signal?: AbortSignal) => unwrap(api.GET("/v1/scans/latest/movers", { params: { query: { limit: 5 } }, signal })),
  journal: (signal?: AbortSignal) => unwrap(api.GET("/v1/journal/insights", { signal })),
  referrals: (signal?: AbortSignal) => unwrap(api.GET("/v1/me/referrals", { signal })),
};

export const emailPrefs = {
  get: (signal?: AbortSignal) => unwrap(api.GET("/v1/me/email-preferences", { signal })),
  setAlerts: (on: boolean) => unwrap(api.PATCH("/v1/me/email-preferences", { body: { alerts: on } })),
};

/** Split pasted text into ticker candidates (the server decides which are valid). */
export function parseTickers(text: string, max = 200): string[] {
  const seen = new Set<string>();
  for (const raw of text.split(/[\s,;]+/)) {
    const t = raw.trim().toUpperCase().replace(/^\$/, "");
    if (t) seen.add(t);
    if (seen.size >= max) break;
  }
  return [...seen];
}
