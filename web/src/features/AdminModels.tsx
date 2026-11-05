"use client";

// Admin > Models (admins only): the ML v4 model registry, forward shadow results and
// monitoring. Moving a model between stages goes through the API, which builds the
// evidence itself (stored evaluation, promotion gates) and records who did it and why.
// Nothing here changes what customers see: a registry champion is not wired into AI Confidence.
import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import { Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { frac } from "@/features/AdminAnalytics";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";
import { MlNextSteps } from "@/features/MlNextSteps";

type Status = "candidate" | "validated" | "shadow" | "champion" | "retired";
type Target = Exclude<Status, "candidate">;
type Gate = { gate: string; value: unknown; threshold: unknown; passed: boolean; note?: string };
type Gates = { gates_version: string; passed: boolean; checks: Gate[] };
type Block = { n?: number; roc_auc?: number | null; brier?: number | null; ece?: number | null };
type Model = {
  id: number; model_version: string; provider: string; kind: string; feature_set: string | null;
  target_label: string; horizon_days: number; dataset_version: string | null; status: Status;
  status_changed_at: string | null; trained_at: string | null; created_at: string | null; has_artifact: boolean;
  artifact_sha256: string | null;
  metrics: { evaluation_version?: string; walk_forward_raw?: Block; walk_forward_calibrated?: Block;
    holdout?: { raw?: Block; calibrated?: Block }; vs_ai_confidence_as_served?: { walk_forward?: { delta?: number | null } } };
  calibration: { method?: string } | null;
};
type Shadow = {
  scored: number; labeled: number; entry_days: number; last_scored_at: string | null; hit_rate?: number | null;
  roc_auc?: number | null; brier?: number | null; ece?: number | null; note?: string;
  vs_ai_confidence_as_served?: { rows: number; model_auc: number | null; served_auc: number | null; auc_delta: number | null; brier_delta: number | null };
};
type Event = { id: number; from_status: string | null; to_status: string; actor: string; reason: string | null; created_at: string };
type Check = { check: string; status: string; value: unknown; detail: string };
type Monitoring = { status: string; generated_at: string; models: { model_id: number; model_version: string; status: string; checks: Check[] }[] };
type ShadowView = { models: { model: Pick<Model, "id" | "model_version" | "provider" | "feature_set" | "status">; shadow: Shadow; promotion_gates: Gates | null }[];
  disagreement: { target_label: string; horizon_days: number; models: number; rows: number; mean_range?: number; share_range_over_0_10?: number; note?: string }[] };

const STATUSES: Status[] = ["candidate", "validated", "shadow", "champion", "retired"];
const NEXT: Record<Status, Target[]> = { candidate: ["validated", "retired"], validated: ["shadow", "retired"], shadow: ["champion", "retired"], champion: ["retired"], retired: [] };
const VERB: Record<Status, string> = { candidate: "", validated: "Mark validated", shadow: "Start shadow test", champion: "Make champion", retired: "Retire" };
const num = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
const signed = (v: number | null | undefined, d = 3) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}`);
const show = (v: unknown) => (v == null ? "—" : typeof v === "number" ? String(Math.round(v * 10000) / 10000) : String(v));

function StagePill({ status }: { status: string }) {
  const tone = status === "champion" ? "gold" : status === "shadow" ? "up" : status === "retired" ? undefined : "warn";
  return <Pill tone={tone}>{status}</Pill>;
}

function HealthPill({ status }: { status: string }) {
  const tone = status === "OK" ? "up" : status === "WARNING" ? "down" : "warn";
  return <Pill tone={tone}>{status === "WAITING" ? "Waiting for data" : status.toLowerCase().replace(/^./, (c) => c.toUpperCase())}</Pill>;
}

function GateList({ gates }: { gates: Gates }) {
  return (
    <div className="table-wrap"><table className="table">
      <caption className="cap">Promotion gates ({gates.gates_version}): {gates.passed ? "all passed" : `${gates.checks.filter((g) => !g.passed).length} not yet passed`}</caption>
      <thead><tr><th scope="col">Gate</th><th scope="col">Value</th><th scope="col">Needs</th><th scope="col">Result</th></tr></thead>
      <tbody>{gates.checks.map((g) => (
        <tr key={g.gate}><th scope="row">{g.gate.replaceAll("_", " ")}{g.note ? <div className="cap">{g.note}</div> : null}</th>
          <td className="mono">{show(g.value)}</td><td className="mono">{show(g.threshold)}</td><td><Pill tone={g.passed ? "up" : "warn"}>{g.passed ? "Pass" : "Not yet"}</Pill></td></tr>
      ))}</tbody>
    </table></div>
  );
}

function MetricRow({ label, b }: { label: string; b?: Block }) {
  return <tr><th scope="row">{label}</th><td className="num mono">{b?.n ?? "—"}</td><td className="num mono">{num(b?.roc_auc)}</td><td className="num mono">{num(b?.brier)}</td><td className="num mono">{num(b?.ece)}</td></tr>;
}

function ModelDetail({ id, onClose, onChanged }: { id: number; onClose: () => void; onChanged: () => void }) {
  const d = useApi(`admin-ml-model:${id}`, (signal) => unwrap(api.GET("/v1/admin/ml/models/{model_id}", { params: { path: { model_id: id } }, signal })) as unknown as Promise<{ model: Model; events: Event[]; shadow: Shadow; promotion_gates: Gates | null }>);
  const [to, setTo] = useState<Target | null>(null);
  const [reason, setReason] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const { busy, error, run, clear } = useAction();
  const m = d.data?.model;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!to) return;
    const res = await run(() => unwrap(api.POST("/v1/admin/ml/models/{model_id}/transition", { params: { path: { model_id: id } }, body: { to_status: to, reason, approve: to === "champion" } })));
    if (res !== undefined) { setDone(`Moved to ${to}.`); setTo(null); setReason(""); d.reload(); onChanged(); }
  };
  let body: ReactNode;
  if (d.error && !d.data) body = <ErrorState error={d.error} onRetry={d.reload} what="this model" />;
  else if (!d.data || !m) body = <Skeleton rows={6} label="Loading the model" />;
  else {
    const s = d.data.shadow;
    body = (
      <div className="stack-sm">
        <p><StagePill status={m.status} /> <span className="cap">{m.provider} · {m.feature_set} · {m.target_label} at {m.horizon_days} days · trained on {m.dataset_version ?? "—"}</span></p>
        <p className="cap">Model file {m.has_artifact ? `stored (sha256 ${m.artifact_sha256?.slice(0, 12)}…)` : "missing"} · calibration {m.calibration?.method ?? "none"} · evaluation {m.metrics.evaluation_version ?? "not attached"}</p>
        <h3 className="h3">Evaluation (before shadow)</h3>
        {m.metrics.evaluation_version ? (
          <div className="table-wrap"><table className="table">
            <thead><tr><th scope="col">Sample</th><th scope="col" className="num">Rows</th><th scope="col" className="num">AUC</th><th scope="col" className="num">Brier</th><th scope="col" className="num">ECE</th></tr></thead>
            <tbody>
              <MetricRow label="Walk-forward (raw)" b={m.metrics.walk_forward_raw} />
              <MetricRow label="Walk-forward (calibrated)" b={m.metrics.walk_forward_calibrated} />
              <MetricRow label="Final holdout (raw)" b={m.metrics.holdout?.raw} />
              <MetricRow label="Final holdout (calibrated)" b={m.metrics.holdout?.calibrated} />
            </tbody>
          </table></div>
        ) : <p className="cap">No evaluation stored yet. Register models from a training run with register=true to attach one.</p>}
        {m.metrics.vs_ai_confidence_as_served?.walk_forward?.delta != null && <p className="cap">Walk-forward AUC vs AI Confidence as served: {signed(m.metrics.vs_ai_confidence_as_served.walk_forward.delta)}</p>}
        <h3 className="h3">Forward shadow test</h3>
        {s.scored === 0 ? <p className="cap">No shadow predictions yet{m.status === "shadow" ? "; the shadow job runs weekdays after the close" : ""}.</p> : (
          <>
            <p className="cap">{s.scored} scored · {s.labeled} matured over {s.entry_days} trading days · last scored {etTime(s.last_scored_at)}</p>
            {s.labeled > 0 && <p className="cap">Hit rate {frac(s.hit_rate)} · AUC {num(s.roc_auc)} · Brier {num(s.brier)} · ECE {num(s.ece)}</p>}
            {s.vs_ai_confidence_as_served && <p className="cap">Same {s.vs_ai_confidence_as_served.rows} rows: AUC {num(s.vs_ai_confidence_as_served.model_auc)} vs AI Confidence {num(s.vs_ai_confidence_as_served.served_auc)} ({signed(s.vs_ai_confidence_as_served.auc_delta)})</p>}
          </>
        )}
        {d.data.promotion_gates && <GateList gates={d.data.promotion_gates} />}
        <h3 className="h3">History</h3>
        <ol className="stack-sm">{d.data.events.map((e) => (
          <li key={e.id} className="cap">{etTime(e.created_at)} · {e.from_status ?? "new"} → <strong>{e.to_status}</strong> by {e.actor}{e.reason ? `: ${e.reason}` : ""}</li>
        ))}</ol>
        {NEXT[m.status].length > 0 && (
          <form className="stack-sm" onSubmit={submit}>
            <h3 className="h3">Change stage</h3>
            <div className="inline-form">{NEXT[m.status].map((s2) => (
              <button key={s2} type="button" className={`btn btn-sm${to === s2 ? " btn-primary" : ""}${s2 === "retired" ? " btn-danger" : ""}`} aria-pressed={to === s2} onClick={() => { setTo(s2); clear(); setDone(null); }}>{VERB[s2]}</button>
            ))}</div>
            {to && (
              <>
                <label className="field"><span>Reason (kept in the history)</span><textarea required minLength={3} maxLength={500} value={reason} onChange={(e) => setReason(e.target.value)} /></label>
                {to === "champion" && <p className="notice">Making this model champion retires the current champion for {m.target_label} at {m.horizon_days} days. It does not change AI Confidence or anything customers see.</p>}
                <button type="submit" className="btn btn-primary" disabled={busy || reason.trim().length < 3}>{busy ? "Saving…" : `Confirm: ${VERB[to].toLowerCase()}${to === "champion" ? " (I approve)" : ""}`}</button>
              </>
            )}
            <ErrorLine error={error} />
          </form>
        )}
        {done && <p className="notice" role="status">{done}</p>}
      </div>
    );
  }
  return <Dialog open title={m ? m.model_version : "Model"} onClose={onClose}>{body}</Dialog>;
}

function Registry() {
  const [status, setStatus] = useState<Status | "">("");
  const [open, setOpen] = useState<number | null>(null);
  const list = useApi(`admin-ml-models:${status}`, (signal) => unwrap(api.GET("/v1/admin/ml/models", { params: { query: status ? { status } : {} }, signal })) as unknown as Promise<{ models: Model[]; count: number }>);
  return (
    <Card title="Model registry" id="ml-registry" aside={list.data ? `${list.data.count} models` : undefined}>
      <p className="cap">Every trained ML v4 model and ensemble, its stage and evidence. A champion here does not change what customers see.</p>
      <label className="field"><span>Stage</span>
        <select value={status} onChange={(e) => setStatus(e.target.value as Status | "")}>
          <option value="">All stages</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </label>
      {list.error && !list.data ? <ErrorState error={list.error} onRetry={list.reload} what="the model registry" />
        : !list.data ? <Skeleton rows={4} label="Loading the model registry" />
        : list.data.models.length === 0 ? <Empty title="No models registered yet.">Models appear after a training run with register=true (ML v4 Train &amp; Evaluate workflow).</Empty>
        : (
          <div className="table-wrap"><table className="table">
            <thead><tr><th scope="col">Model</th><th scope="col">Stage</th><th scope="col" className="hide-narrow">Target</th><th scope="col" className="num hide-narrow">Walk-forward AUC</th><th scope="col" className="hide-narrow">Updated</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>{list.data.models.map((m) => (
              <tr key={m.id}>
                <td><strong className="mono">{m.provider}</strong> <span className="cap">{m.feature_set}</span><div className="cap mono">{m.model_version}</div></td>
                <td><StagePill status={m.status} /></td>
                <td className="cap hide-narrow">{m.target_label} · {m.horizon_days}d</td>
                <td className="num mono hide-narrow">{num(m.metrics.walk_forward_raw?.roc_auc)}</td>
                <td className="cap hide-narrow">{etTime(m.status_changed_at)}</td>
                <td className="num"><button type="button" className="btn btn-sm" aria-label={`Open ${m.model_version}`} onClick={() => setOpen(m.id)}>Open</button></td>
              </tr>
            ))}</tbody>
          </table></div>
        )}
      {open != null && <ModelDetail id={open} onClose={() => setOpen(null)} onChanged={list.reload} />}
    </Card>
  );
}

function ShadowResults() {
  const [days, setDays] = useState(30);
  const v = useApi(`admin-ml-shadow:${days}`, (signal) => unwrap(api.GET("/v1/admin/ml/shadow", { params: { query: { days } }, signal })) as unknown as Promise<ShadowView>);
  return (
    <Card title="Shadow test" id="ml-shadow">
      <p className="cap">Forward predictions from models in shadow, scored weekdays after the close and never shown to customers. Compared with AI Confidence as customers saw it, on the same matured signals.</p>
      <label className="field"><span>Period</span>
        <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
          {[[30, "30 days"], [90, "90 days"], [180, "180 days"]].map(([d, l]) => <option key={d} value={d}>{l}</option>)}
        </select>
      </label>
      {v.error && !v.data ? <ErrorState error={v.error} onRetry={v.reload} what="shadow results" />
        : !v.data ? <Skeleton rows={3} label="Loading shadow results" />
        : v.data.models.length === 0 ? <Empty title="No model is in shadow yet.">Open a validated model in the registry and start its shadow test.</Empty>
        : (
          <>
            <div className="table-wrap"><table className="table">
              <thead><tr><th scope="col">Model</th><th scope="col" className="num">Scored</th><th scope="col" className="num">Matured</th><th scope="col" className="num">AUC</th><th scope="col" className="num">vs AI Confidence</th><th scope="col">Gates</th></tr></thead>
              <tbody>{v.data.models.map(({ model, shadow, promotion_gates }) => (
                <tr key={model.id}>
                  <td><strong className="mono">{model.provider}</strong> <span className="cap">{model.feature_set}</span> <StagePill status={model.status} /></td>
                  <td className="num mono">{shadow.scored}</td><td className="num mono">{shadow.labeled}</td>
                  <td className="num mono">{num(shadow.roc_auc)}</td><td className="num mono">{signed(shadow.vs_ai_confidence_as_served?.auc_delta)}</td>
                  <td>{promotion_gates ? <Pill tone={promotion_gates.passed ? "up" : "warn"}>{promotion_gates.passed ? "Passed" : `${promotion_gates.checks.filter((g) => g.passed).length}/${promotion_gates.checks.length}`}</Pill> : <span className="cap">—</span>}</td>
                </tr>
              ))}</tbody>
            </table></div>
            {v.data.disagreement.filter((g) => g.rows > 0).map((g) => (
              <p key={`${g.target_label}-${g.horizon_days}`} className="cap">Model disagreement ({g.target_label}, {g.horizon_days}d, {g.models} models, {g.rows} shared signals): average spread {frac(g.mean_range)}; {frac(g.share_range_over_0_10)} differ by more than 10 points. {g.note}</p>
            ))}
          </>
        )}
    </Card>
  );
}

function MonitoringCard() {
  const v = useApi("admin-ml-monitoring", (signal) => unwrap(api.GET("/v1/admin/ml/monitoring", { signal })) as unknown as Promise<Monitoring>);
  return (
    <Card title="Monitoring" id="ml-monitoring" aside={v.data ? `Checked ${etTime(v.data.generated_at)}` : undefined}>
      {v.error && !v.data ? <ErrorState error={v.error} onRetry={v.reload} what="model monitoring" />
        : !v.data ? <Skeleton rows={3} label="Loading model monitoring" />
        : v.data.models.length === 0 ? <p className="cap">Nothing to monitor: no model is in shadow or champion.</p>
        : (
          <div className="stack-sm">
            <p><HealthPill status={v.data.status} /> <span className="cap">Shadow job freshness, labels filling in, probability drift, recent calibration, ranking vs AI Confidence and feature schema.</span></p>
            {v.data.models.map((m) => (
              <details key={m.model_id} open={m.status === "WARNING"}>
                <summary><span className="mono">{m.model_version}</span> <HealthPill status={m.status} /></summary>
                <ul className="stack-sm">{m.checks.map((c) => <li key={c.check} className="cap"><HealthPill status={c.status} /> <strong>{c.check.replaceAll("_", " ")}</strong>: {c.detail}</li>)}</ul>
              </details>
            ))}
          </div>
        )}
    </Card>
  );
}

type Production = {
  configured_model: { model_version: string; trained_at: string | null; dataset_version: string | null } | null;
  latest_capture: { signal_at: string; inferred_at: string | null; artifact_sha256: string | null; model_version: string | null } | null;
};

function ModelSummary() {
  const production = useApi("admin-ml-production", (signal) => unwrap(api.GET("/v1/admin/ml/production", { signal })) as unknown as Promise<Production>);
  const registry = useApi("admin-ml-summary", (signal) => unwrap(api.GET("/v1/admin/ml/models", { signal })) as unknown as Promise<{ models: Model[] }>);
  return <Card title="What is trained and running?">
    {production.error ? <ErrorState error={production.error} onRetry={production.reload} what="production confidence evidence" /> : !production.data ? <Skeleton rows={2} label="Loading production confidence" /> : <>
      <h3 className="h3">Production AI Confidence</h3>
      <p>Configured database model: <strong>{production.data.configured_model?.model_version ?? "No active database model recorded"}</strong>.</p>
      <p className="cap">Trained: {etTime(production.data.configured_model?.trained_at)} · dataset: {production.data.configured_model?.dataset_version ?? "Not recorded"}.</p>
      <p>Latest recorded confidence: {production.data.latest_capture ? `${production.data.latest_capture.model_version ?? "Model version not recorded"} · signal ${etTime(production.data.latest_capture.signal_at)}` : "No saved confidence found"}.</p>
      <p className="cap">Inference time: {etTime(production.data.latest_capture?.inferred_at)} · artifact checksum: <span className="mono">{production.data.latest_capture?.artifact_sha256 ?? "Not recorded on this signal"}</span>.</p>
      <p className="cap">The configured model and latest recorded prediction are separate evidence. A local fallback may be used; an unrecorded value is unknown.</p>
    </>}
    <h3 className="h3">ML v4 challengers</h3>
    {registry.error ? <ErrorState error={registry.error} onRetry={registry.reload} what="challenger training records" /> : !registry.data ? <Skeleton rows={3} label="Loading challengers" /> : <div className="table-wrap"><table className="table">
      <thead><tr><th scope="col">Model</th><th scope="col">Training evidence</th><th scope="col">Dataset</th><th scope="col">Evaluation</th><th scope="col">Stage / app use</th></tr></thead>
      <tbody>{["xgboost", "lightgbm", "catboost"].map((provider) => {
        const models = registry.data!.models.filter((m) => m.provider === provider);
        if (!models.length) return <tr key={provider}><th scope="row">{provider}</th><td colSpan={4}>No registered training record. An unregistered workflow artifact may still exist.</td></tr>;
        return models.map((m) => <tr key={m.id}><th scope="row">{provider}<div className="cap">{m.model_version}</div></th>
          <td>{m.trained_at ? etTime(m.trained_at) : "Training time not recorded"}<div className="cap">Artifact: {m.has_artifact ? "Registered" : "Not recorded"}</div></td>
          <td>{m.dataset_version ?? "Not recorded"}</td><td>{m.metrics?.evaluation_version ? "Report recorded" : "No report recorded"}</td>
          <td>{m.status}<div className="cap">{["shadow", "champion"].includes(m.status) ? "Eligible for scheduled shadow scoring" : "Not enabled for shadow scoring"}. Not connected to production AI Confidence.</div></td></tr>);
      })}</tbody></table></div>}
  </Card>;
}

export function ModelsTab() {
  return (
    <div className="stack">
      <p className="notice">ML v4 challengers (XGBoost, LightGBM, CatBoost and ensembles) are tested here first. AI Confidence keeps using the production model until a separate, explicit decision wires a champion in.</p>
      <ModelSummary />
      <MlNextSteps />
      <MonitoringCard />
      <ShadowResults />
      <Registry />
    </div>
  );
}
