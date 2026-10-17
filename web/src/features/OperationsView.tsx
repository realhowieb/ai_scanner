"use client";

import { useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, ErrorLine } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";

export function OperationsView() {
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [problems, setProblems] = useState(true);
  const ops = useApi(`operations:${query}:${problems}`, (signal) => unwrap(api.GET("/v1/admin/requests", {
    signal, params: { query: { q: query, problems_only: problems } },
  })));
  return <Card title="API operations" id="api-operations">
    <form className="row-actions" onSubmit={(e) => { e.preventDefault(); setQuery(q); ops.reload(); }}>
      <label className="field">Support code, route or status<input maxLength={100} value={q} onChange={(e) => setQ(e.target.value)} placeholder="web-… or /v1/scans" /></label>
      <label><input type="checkbox" checked={problems} onChange={(e) => setProblems(e.target.checked)} /> Errors and requests over 1 second</label>
      <button className="btn btn-sm" disabled={ops.loading}>Search / refresh</button>
    </form>
    <ErrorLine error={ops.error} />
    {ops.data && <div className="stack-sm">
      <p className="cap">{ops.data.scope}</p>
      <p>{ops.data.total} requests · {ops.data.errors} server errors · {ops.data.slow} slow requests</p>
      <p className="cap">Scan queue: {ops.data.scan_queue == null ? "unavailable" : ops.data.scan_queue.length ? ops.data.scan_queue.map((s) => `${s.count} ${s.status}`).join(" · ") : "empty"}</p>
      <ul className="stack-sm">{ops.data.requests.map((r) => <li key={`${r.request_id}:${r.at}`}>
        <strong>{r.status} · {r.method} {r.route}</strong> · {r.ms} ms
        <p className="cap">{r.request_id} · {etTime(r.at)}</p>
      </li>)}</ul>
      {ops.data.requests.length === 0 && <p className="cap">No matching requests in this process.</p>}
      <h3 className="strong">Recent provider checks</h3>
      <ul className="stack-sm">{ops.data.providers.map((p, i) => <li key={i}>{p.name}: {p.ok ? "OK" : "Needs attention"} · {p.ms} ms · {etTime(p.at)}</li>)}</ul>
      {!ops.data.providers.length && <p className="cap">Run Services checks below to record provider health.</p>}
    </div>}
  </Card>;
}
