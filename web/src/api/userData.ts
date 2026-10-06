// Watchlist and alert calls, through the typed client.
import { api, unwrap } from "./client";
import type { Schemas } from "./client";

export type Watchlist = Schemas["Watchlist"];
export type WatchlistDetail = Schemas["WatchlistDetail"];
export type Alert = Schemas["Alert"];
export type AlertType = Schemas["AlertType"];
export type AlertCreate = Omit<Schemas["AlertCreate"], "watchlist_only"> & { watchlist_only?: boolean };

const wid = (id: number) => ({ params: { path: { watchlist_id: id } } });

export const watchlists = {
  list: (signal?: AbortSignal) => unwrap(api.GET("/v1/watchlists", { signal })),
  get: (id: number, signal?: AbortSignal) => unwrap(api.GET("/v1/watchlists/{watchlist_id}", { ...wid(id), signal })),
  create: (name: string, makeDefault = false) =>
    unwrap(api.POST("/v1/watchlists", { body: { name, make_default: makeDefault } })),
  rename: (id: number, name: string) =>
    unwrap(api.PATCH("/v1/watchlists/{watchlist_id}", { ...wid(id), body: { name, make_default: false } })),
  makeDefault: (id: number) =>
    unwrap(api.PATCH("/v1/watchlists/{watchlist_id}", { ...wid(id), body: { make_default: true } as Schemas["WatchlistUpdate"] })),
  remove: (id: number) => unwrap(api.DELETE("/v1/watchlists/{watchlist_id}", wid(id))),
  addTickers: (id: number, tickers: string[]) =>
    unwrap(api.POST("/v1/watchlists/{watchlist_id}/tickers", { ...wid(id), body: { tickers } })),
  removeTicker: (id: number, ticker: string) =>
    unwrap(api.DELETE("/v1/watchlists/{watchlist_id}/tickers/{ticker}", { params: { path: { watchlist_id: id, ticker } } })),
  setNote: (id: number, ticker: string, note: string) =>
    unwrap(api.PATCH("/v1/watchlists/{watchlist_id}/tickers/{ticker}", {
      params: { path: { watchlist_id: id, ticker } }, body: { note: note.trim() ? note : null },
    })),
};

export const alerts = {
  list: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts", { signal })),
  types: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts/types", { signal })),
  events: (signal?: AbortSignal) => unwrap(api.GET("/v1/alerts/events", { params: { query: { limit: 20 } }, signal })),
  create: (body: AlertCreate) => unwrap(api.POST("/v1/alerts", { body: body as Schemas["AlertCreate"] })),
  setEnabled: (id: number, enabled: boolean) =>
    unwrap(api.PATCH("/v1/alerts/{alert_id}", { params: { path: { alert_id: id } }, body: { enabled } })),
  remove: (id: number) => unwrap(api.DELETE("/v1/alerts/{alert_id}", { params: { path: { alert_id: id } } })),
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
