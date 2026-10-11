// An in-memory stand-in for the HSF API's watchlist and alert routes, following the
// real rules (duplicates 409, limits 403, invalid tickers reported, owner-only 404),
// with failures injectable per route. Test-only.
import types from "./fixtures/alert-types.json";
import { jsonResponse } from "./helpers";

type Item = { ticker: string; added_at: string; price_when_added: number | null; note: string | null };
type WL = { id: number; name: string; is_default: boolean; items: Item[] };
type AlertRow = { id: number; type: string; ticker: string | null; threshold: number | null; direction: string | null;
  watchlist_only: boolean; enabled: boolean; last_fired_at: string | null; created_at: string };

const TICKER = /^[A-Z0-9][A-Z0-9.-]{0,9}$/;

export function fakeApi(opts: { alertLimit?: number; emailEnabled?: boolean; scan?: { ticker: string; score: number; last: number }[];
  /** The current API: GET /v1/watchlists/{id} carries scan_at and items[].latest. Off = an API that predates them. */
  scanState?: boolean;
  /** Serve GET /v1/watchlists/{id}/intelligence with these per-ticker facts (else 404, like an older API). */
  intel?: Record<string, Record<string, unknown>>;
  /** Serve the alert-rule routes (an in-memory rule list). */
  rules?: boolean;
  /** GET /v1/alerts/events returns these (newest first). */
  events?: Record<string, unknown>[] } = {}) {
  let nextId = 1;
  const lists: WL[] = [];
  const alertRows: AlertRow[] = [];
  const prefs = { digest: true, evening: true, alerts: true };
  const calls: { method: string; path: string; body: unknown }[] = [];
  const failures: { method: string; re: RegExp; status: number; body: unknown; headers?: Record<string, string> }[] = [];
  const limit = opts.alertLimit ?? 5;
  const rules: { id: number; watchlist_id: number | null; rule_type: string; threshold: number | null; [k: string]: unknown }[] = [];
  const ruleTypes = [
    { type: "HSF_SCORE_CROSS_ABOVE", label: "HSF Score crosses above", description: "Fires once when the HSF Score moves above your value.",
      operator: "crosses_above", kind: "transition", threshold: { min: 0, max: 100, default: 80 }, takes_value: false, default_cooldown_seconds: 3600, available: true },
    { type: "SETUP_APPEARED", label: "New HSF setup", description: "Fires when the ticker becomes a ranked HSF setup.",
      operator: "appears", kind: "transition", threshold: null, takes_value: true, default_cooldown_seconds: 3600, available: true },
    { type: "PREBREAKOUT_ACTIVE", label: "Becomes PreBreakout", description: "Premium.", operator: "becomes_true", kind: "transition",
      threshold: null, takes_value: false, default_cooldown_seconds: 3600, available: false },
  ];
  const scanAt = new Date().toISOString();

  const summary = (w: WL) => ({ id: w.id, name: w.name, is_default: w.is_default, symbol_count: w.items.length });
  const detail = (w: WL) => ({ ...summary(w), items: w.items });
  const err = (status: number, detail: string) => jsonResponse({ detail }, status, { "x-request-id": `srv-${status}-${calls.length}` });

  async function handle(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname.replace(/^\/api\/hsf/, "");
    const method = req.method;
    const text = method === "GET" || method === "DELETE" ? "" : await req.text();
    const body = text ? JSON.parse(text) : null;
    calls.push({ method, path, body });
    const f = failures.findIndex((x) => x.method === method && x.re.test(path));
    if (f >= 0) {
      const [x] = failures.splice(f, 1);
      return jsonResponse(x!.body, x!.status, { "x-request-id": "srv-injected", ...(x!.headers ?? {}) });
    }
    let m: RegExpMatchArray | null;
    if (path === "/v1/scans/latest") {
      return jsonResponse({ scan_at: scanAt, total: (opts.scan ?? []).length, max_results: 100, limited: false,
        setups: (opts.scan ?? []).map((s) => ({ ticker: s.ticker, score: s.score, last: s.last, primary_setup: "breakout", status: "STRONG",
          n_signals: 1, chg_pct: 1, gap_pct: null, rvol: null, prob: null, signals: [], fading: false, breakout_score: null })) });
    }
    if (opts.intel && (m = path.match(/^\/v1\/watchlists\/(\d+)\/intelligence$/))) {
      const w = lists.find((x) => x.id === Number(m![1]));
      if (!w) return err(404, "No such watchlist.");
      return jsonResponse({ watchlist_id: w.id, name: w.name, market_session: "open", scan_available: true, last_scan_at: scanAt,
        previous_scan_at: scanAt, market_data_as_of: scanAt, stale: false, scan_total: 2, prebreakout_locked: false,
        coverage: { symbols: w.items.length, enriched: w.items.length, missing: 0 }, unavailable_fields: ["company_name", "price_change", "rsi"],
        items: w.items.map((i) => ({ ticker: i.ticker, signals: [], ranked: true, in_latest_scan: true, freshness: "fresh",
          active_alert_count: rules.filter((r) => r.watchlist_id === w.id).length, ...(opts.intel![i.ticker] ?? {}) })) });
    }
    if (opts.rules && path === "/v1/alerts/rules/types") return jsonResponse(ruleTypes);
    if (opts.rules && path === "/v1/alerts/rules" && method === "GET") return jsonResponse({ limit, used: rules.length + alertRows.length,
      capabilities: { tier: "pro", max_watchlists: 50, max_symbols_per_watchlist: null, max_symbols_per_request: 200, max_active_alerts: limit,
        alert_rule_types: ruleTypes.filter((t) => t.available).map((t) => t.type), delivery_channels: ["in_app", "email"] }, rules });
    if (opts.rules && path === "/v1/alerts/rules" && method === "POST") {
      if (rules.length + alertRows.length >= limit) return err(403, `You've reached the maximum of ${limit} active alerts on your plan.`);
      const spec = ruleTypes.find((t) => t.type === body.rule_type)!;
      const r = { id: nextId++, watchlist_id: body.watchlist_id ?? null, ticker: body.ticker ?? null, rule_type: body.rule_type,
        operator: spec.operator, threshold: body.threshold ?? null, value: null, enabled: true, delivery_channels: ["in_app"],
        cooldown_seconds: spec.default_cooldown_seconds, created_at: scanAt, updated_at: scanAt, last_evaluated_at: null, last_triggered_at: null };
      rules.push(r);
      return jsonResponse(r, 201);
    }
    if (opts.rules && (m = path.match(/^\/v1\/alerts\/rules\/(\d+)$/)) && method === "DELETE") {
      const i = rules.findIndex((r) => r.id === Number(m![1]));
      if (i < 0) return err(404, "No such alert rule.");
      rules.splice(i, 1);
      return new Response(null, { status: 204 });
    }
    if (path === "/v1/watchlists" && method === "GET") return jsonResponse(lists.map(summary));
    if (path === "/v1/watchlists" && method === "POST") {
      if (lists.some((w) => w.name.toLowerCase() === body.name.toLowerCase())) return err(409, `You already have a watchlist named "${body.name}".`);
      if (lists.length >= 50) return err(403, "You've reached the limit of 50 watchlists.");
      const w: WL = { id: nextId++, name: body.name, is_default: lists.length === 0 || !!body.make_default, items: [] };
      if (w.is_default) lists.forEach((x) => (x.is_default = false));
      lists.push(w);
      return jsonResponse(detail(w), 201);
    }
    if ((m = path.match(/^\/v1\/watchlists\/(\d+)(?:\/tickers(?:\/([^/]+))?)?$/))) {
      const w = lists.find((x) => x.id === Number(m![1]));
      if (!w) return err(404, "Watchlist not found.");
      const sub = path.includes("/tickers");
      const t = m[2] ? decodeURIComponent(m[2]) : null;
      if (!sub && method === "GET") {
        if (!opts.scanState) return jsonResponse(detail(w));
        const rows = opts.scan ?? [];
        return jsonResponse({ ...detail(w), scan_at: scanAt, scan_total: rows.length, stale: false,
          items: w.items.map((i) => {
            const k = rows.findIndex((r) => r.ticker === i.ticker);
            const r = rows[k];
            return { ...i, latest: r ? { ticker: r.ticker, score: r.score, last: r.last, primary_setup: "breakout", status: "STRONG", n_signals: 1,
              chg_pct: 1.5, gap_pct: null, rvol: null, prob: null, signals: [], fading: false, breakout_score: null, rank: k + 1 } : null };
          }) });
      }
      if (!sub && method === "PATCH") {
        if (body.name && lists.some((x) => x !== w && x.name.toLowerCase() === body.name.toLowerCase())) return err(409, `You already have a watchlist named "${body.name}".`);
        if (body.name) w.name = body.name;
        if (body.make_default) { lists.forEach((x) => (x.is_default = false)); w.is_default = true; }
        return jsonResponse(detail(w));
      }
      if (!sub && method === "DELETE") { lists.splice(lists.indexOf(w), 1); return new Response(null, { status: 204 }); }
      if (sub && !t && method === "POST") {
        const out = { added: [] as string[], already_present: [] as string[], invalid: [] as string[] };
        for (const raw of body.tickers as string[]) {
          const s = raw.toUpperCase();
          if (!TICKER.test(s)) out.invalid.push(raw);
          else if (w.items.some((i) => i.ticker === s)) out.already_present.push(s);
          else { w.items.push({ ticker: s, added_at: new Date().toISOString(), price_when_added: null, note: null }); out.added.push(s); }
        }
        return jsonResponse(out);
      }
      const it = w.items.find((i) => i.ticker === t);
      if (!it) return err(404, "Ticker not in this watchlist.");
      if (method === "DELETE") { w.items.splice(w.items.indexOf(it), 1); return new Response(null, { status: 204 }); }
      if (method === "PATCH") { it.note = body.note; return new Response(null, { status: 204 }); }
    }
    if (path === "/v1/alerts/types") return jsonResponse(types);
    if (path === "/v1/alerts/events") return jsonResponse(opts.events ?? []);
    if (path === "/v1/alerts" && method === "GET") return jsonResponse({ limit, used: alertRows.length, email_enabled: opts.emailEnabled ?? true, alerts: alertRows });
    if (path === "/v1/alerts" && method === "POST") {
      const spec = types.find((x) => x.type === body.type);
      if (!spec) return err(422, "Unknown alert type.");
      if (spec.threshold && (body.threshold == null || (spec.threshold.min_exclusive ? body.threshold <= spec.threshold.min : body.threshold < spec.threshold.min)))
        return err(422, `Threshold must be ${spec.threshold.min_exclusive ? "greater than" : "at least"} ${spec.threshold.min}.`);
      if (alertRows.length >= limit) return err(403, `Your plan allows a maximum of ${limit} alert${limit === 1 ? "" : "s"} on this plan.`);
      if (alertRows.some((a) => a.type === body.type && a.ticker === (body.ticker ?? null) && a.threshold === (body.threshold ?? null) && a.direction === (body.direction ?? null)))
        return err(409, "You already have this alert.");
      const a: AlertRow = { id: nextId++, type: body.type, ticker: body.ticker ?? null, threshold: body.threshold ?? null, direction: body.direction ?? null,
        watchlist_only: !!body.watchlist_only, enabled: true, last_fired_at: null, created_at: new Date().toISOString() };
      alertRows.push(a);
      return jsonResponse(a, 201);
    }
    if ((m = path.match(/^\/v1\/alerts\/(\d+)$/))) {
      const a = alertRows.find((x) => x.id === Number(m![1]));
      if (!a) return err(404, "Alert not found.");
      if (method === "PATCH") { a.enabled = !!body.enabled; return jsonResponse(a); }
      if (method === "DELETE") { alertRows.splice(alertRows.indexOf(a), 1); return new Response(null, { status: 204 }); }
    }
    if (path === "/v1/me/email-preferences") {
      if (method === "PATCH") Object.assign(prefs, body);
      return jsonResponse(prefs);
    }
    if (path === "/v1/billing/checkout") return jsonResponse({ url: "https://checkout.example/x", mode: "checkout" });
    return err(404, `No fake for ${method} ${path}`);
  }

  return {
    lists, alertRows, rules, calls, prefs,
    fetch: (input: Request | string, init?: RequestInit) => handle(typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input),
    fail: (method: string, re: RegExp, status: number, body: unknown = { detail: "The HSF service is unavailable right now." }, headers?: Record<string, string>) =>
      failures.push({ method, re, status, body, headers }),
    seedList: (name: string, tickers: string[] = [], isDefault = false) => {
      const w: WL = { id: nextId++, name, is_default: isDefault || lists.length === 0, items: tickers.map((t) => ({ ticker: t, added_at: new Date().toISOString(), price_when_added: 10, note: null })) };
      lists.push(w);
      return w;
    },
    count: (method: string, re: RegExp) => calls.filter((c) => c.method === method && re.test(c.path)).length,
  };
}
