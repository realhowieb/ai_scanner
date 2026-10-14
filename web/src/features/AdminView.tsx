"use client";

// Admin console (admins only): accounts, health and maintenance tools, moved from the
// classic app's Admin page. The API checks the admin role on every call; this page only
// hides itself from everyone else.
import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ConfirmDialog, Dialog } from "@/components/Dialog";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

type User = Schemas["AdminUser"];
type Check = Schemas["AdminCheck"];
const PLANS = ["basic", "pro", "premium"] as const;
type Plan = (typeof PLANS)[number];
const TABS = [["users", "Users"], ["health", "Health"], ["tools", "Tools"]] as const;
type Tab = (typeof TABS)[number][0];

const GOOD = new Set(["HEALTHY", "WAITING", "ok"]);
const BAD = new Set(["CRITICAL", "error", "FAILED", "CONSECUTIVE_FAILURES"]);
function StatusPill({ status }: { status: string | null | undefined }) {
  const s = status ?? "UNKNOWN";
  const label = s === "not_configured" ? "Not configured" : s.replaceAll("_", " ").toLowerCase().replace(/^./, (c) => c.toUpperCase());
  return <Pill tone={GOOD.has(s) ? "up" : BAD.has(s) ? "down" : "warn"}>{label}</Pill>;
}

function Done({ message }: { message: string | null }) {
  return message ? <p className="notice" role="status">{message}</p> : null;
}

// ---- users -------------------------------------------------------------------------------------
function ManageUser({ u, self, onClose, onChanged }: { u: User; self: string; onClose: () => void; onChanged: () => void }) {
  const [tier, setTier] = useState<string>(u.tier);
  const [active, setActive] = useState(u.is_active);
  const [role, setRole] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const act = useAction();
  const roleAct = useAction();
  const legacyTier = !(PLANS as readonly string[]).includes(u.tier);
  const changed = tier !== u.tier || active !== u.is_active;
  const isSelf = u.username.toLowerCase() === self.toLowerCase();
  const target = { username: u.username };
  const run = (fn: () => Promise<{ message: string }>) => void act.run(async () => {
    const out = await fn();
    setDone(out.message);
    onChanged();
    return true;
  });
  const save = (e: FormEvent) => {
    e.preventDefault();
    if (!changed) return;
    run(() => unwrap(api.POST("/v1/admin/users/update", { body: { ...target,
      ...(tier !== u.tier ? { tier: tier as Plan } : {}), ...(active !== u.is_active ? { is_active: active } : {}) } })));
  };
  return (
    <Dialog open title={u.username} onClose={onClose}>
      <div className="stack-sm">
        <p className="cap">{u.full_name || "No name"} · email {u.email_verified ? "verified" : "not verified"} · {u.is_admin ? "admin" : "not admin"}</p>
        <form className="stack-sm" onSubmit={save} aria-label="Plan and status">
          <label className="field"><span>Plan</span>
            <select value={tier} onChange={(e) => setTier(e.target.value)}>
              {legacyTier && <option value={u.tier}>{u.tier}</option>}
              {PLANS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <label className="check"><input type="checkbox" checked={active} disabled={isSelf} onChange={(e) => setActive(e.target.checked)} /> Active (can sign in)</label>
          <div className="row-actions end"><button type="submit" className="btn btn-primary" disabled={!changed || act.busy}>{act.busy ? "Saving…" : "Save changes"}</button></div>
        </form>
        {!u.email_verified && (
          <div className="row-actions">
            <button type="button" className="btn btn-sm" disabled={act.busy}
              onClick={() => run(() => unwrap(api.POST("/v1/admin/users/verify", { body: target })))}>Mark email verified</button>
            <button type="button" className="btn btn-sm" disabled={act.busy}
              onClick={() => run(() => unwrap(api.POST("/v1/admin/users/resend-verification", { body: target })))}>Resend verification email</button>
          </div>
        )}
        {!isSelf && (
          <div className="row-actions">
            <button type="button" className={`btn btn-sm${u.is_admin ? " btn-danger-outline" : ""}`} onClick={() => { roleAct.clear(); setRole(true); }}>
              {u.is_admin ? "Revoke admin" : "Make admin"}
            </button>
          </div>
        )}
        <ErrorLine error={act.error} />
        <Done message={done} />
      </div>
      <ConfirmDialog open={role} title={u.is_admin ? `Remove the admin role from ${u.username}?` : `Make ${u.username} an admin?`}
        confirmLabel={u.is_admin ? "Revoke admin" : "Make admin"} busy={roleAct.busy}
        body={<p>{u.is_admin ? "They lose the Admin page and admin-only data." : "They get this Admin page, every account and admin-only data."}</p>}
        error={<ErrorLine error={roleAct.error} />} onClose={() => setRole(false)}
        onConfirm={() => void roleAct.run(async () => {
          const out = await unwrap(api.POST("/v1/admin/users/role", { body: { ...target, is_admin: !u.is_admin } }));
          setRole(false); setDone(out.message); onChanged(); return true;
        })} />
    </Dialog>
  );
}

function CreateUser({ onDone }: { onDone: () => void }) {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [tier, setTier] = useState<Plan>("basic");
  const [done, setDone] = useState<string | null>(null);
  const act = useAction();
  const valid = email.includes("@") && name.trim() !== "" && password.length > 0;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    setDone(null);
    const out = await act.run(() => unwrap(api.POST("/v1/admin/users", { body: { email: email.trim(), full_name: name.trim(), password, tier, active: true } })));
    if (out) { setDone(out.message); setEmail(""); setName(""); setPassword(""); onDone(); }
  };
  return (
    <form className="stack-sm" onSubmit={submit} aria-label="Create an account">
      <div className="inline-form">
        <label className="field"><span>Email</span><input type="email" value={email} autoComplete="off" onChange={(e) => setEmail(e.target.value)} /></label>
        <label className="field"><span>Full name</span><input value={name} autoComplete="off" onChange={(e) => setName(e.target.value)} /></label>
        <label className="field"><span>Password</span><input type="password" value={password} autoComplete="new-password" onChange={(e) => setPassword(e.target.value)} /></label>
        <label className="field"><span>Plan</span>
          <select value={tier} onChange={(e) => setTier(e.target.value as Plan)}>{PLANS.map((p) => <option key={p} value={p}>{p}</option>)}</select>
        </label>
        <button type="submit" className="btn btn-primary" disabled={!valid || act.busy}>{act.busy ? "Creating…" : "Create account"}</button>
      </div>
      <p className="cap">At least 10 characters, same rules as sign-up. Their email starts unverified.</p>
      <ErrorLine error={act.error} />
      <Done message={done} />
    </form>
  );
}

function UsersTab({ self }: { self: string }) {
  const list = useApi("admin-users", (signal) => unwrap(api.GET("/v1/admin/users", { signal })));
  const audit = useApi("admin-audit", (signal) => unwrap(api.GET("/v1/admin/audit", { signal })));
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const reload = () => { list.reload(); audit.reload(); };
  const users = useMemo(() => {
    const t = q.trim().toLowerCase();
    return (list.data?.users ?? []).filter((u) => !t || u.username.toLowerCase().includes(t) || (u.full_name ?? "").toLowerCase().includes(t));
  }, [list.data, q]);
  const selected = list.data?.users.find((u) => u.username === open) ?? null;
  const all = list.data?.users ?? [];
  return (
    <div className="stack">
      <Card title="Accounts" id="accounts" aside={list.data ? `${all.length} accounts · ${all.filter((u) => u.is_admin).length} admins · ${all.filter((u) => !u.email_verified).length} unverified` : undefined}>
        <label className="field"><span>Search</span><input type="search" value={q} placeholder="Email or name" onChange={(e) => setQ(e.target.value)} /></label>
        {list.error && !list.data ? <ErrorState error={list.error} onRetry={list.reload} what="the accounts" />
          : !list.data ? <Skeleton rows={6} label="Loading accounts" />
          : users.length === 0 ? <Empty title="No matching accounts." />
          : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">Account</th><th scope="col">Plan</th><th scope="col">Status</th><th scope="col" className="hide-narrow">Created</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
                <tbody>{users.map((u) => (
                  <tr key={u.username}>
                    <td><strong>{u.username}</strong>{u.full_name && <div className="cap">{u.full_name}</div>}</td>
                    <td>{u.tier}</td>
                    <td>
                      {!u.is_active && <Pill tone="down">Inactive</Pill>} {u.is_admin && <Pill tone="gold">Admin</Pill>} {!u.email_verified && <Pill tone="warn">Unverified</Pill>}
                      {u.is_active && !u.is_admin && u.email_verified && <span className="cap">Active</span>}
                    </td>
                    <td className="cap hide-narrow">{etTime(u.created_at)}</td>
                    <td className="num"><button type="button" className="btn btn-sm" aria-label={`Manage ${u.username}`} onClick={() => setOpen(u.username)}>Manage</button></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
      </Card>
      {selected && <ManageUser key={selected.username} u={selected} self={self} onClose={() => setOpen(null)} onChanged={reload} />}
      <Card title="Create an account" id="create"><CreateUser onDone={reload} /></Card>
      <Card title="Recent admin actions" id="audit">
        {audit.error && !audit.data ? <ErrorState error={audit.error} onRetry={audit.reload} what="the audit log" />
          : !audit.data ? <Skeleton rows={3} label="Loading the audit log" />
          : audit.data.events.length === 0 ? <Empty title="No admin actions recorded yet." />
          : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th scope="col">When</th><th scope="col">Admin</th><th scope="col">Action</th><th scope="col">Account</th><th scope="col" className="hide-narrow">Details</th></tr></thead>
                <tbody>{audit.data.events.map((e, i) => (
                  <tr key={`${e.at}-${i}`}><td className="cap">{etTime(e.at)}</td><td>{e.actor}</td><td>{e.action}</td><td>{e.target}</td><td className="cap hide-narrow">{e.details}</td></tr>
                ))}</tbody>
              </table>
            </div>
          )}
      </Card>
    </div>
  );
}

// ---- health ------------------------------------------------------------------------------------
function Checks({ checks }: { checks: Check[] }) {
  return (
    <ul className="stack-sm" aria-label="Service checks">
      {checks.map((c) => (
        <li key={c.name}><StatusPill status={c.status} /> <strong>{c.name}</strong> <span className="cap">{c.detail}{c.ms != null ? ` · ${Math.round(c.ms)} ms` : ""}</span></li>
      ))}
    </ul>
  );
}

function HealthTab() {
  const h = useApi("admin-health", (signal) => unwrap(api.GET("/v1/admin/health", { signal })));
  const probe = useAction();
  const [providers, setProviders] = useState<Schemas["AdminProviders"] | null>(null);
  const ops = h.data?.operations as { system_status?: string; generated_at?: string | null; stale?: boolean;
    workflows?: { workflow: string; label: string; status: string; last_success: string | null }[] } | undefined;
  return (
    <div className="stack">
      {h.error && !h.data ? <ErrorState error={h.error} onRetry={h.reload} what="system health" />
        : !h.data ? <Skeleton rows={6} label="Loading system health" />
        : (
          <>
            <Card title="System health" id="system" aside={ops?.generated_at ? `Snapshot ${etTime(ops.generated_at)}${ops.stale ? " · stale" : ""}` : "No snapshot yet"}>
              <div className="metric-strip">
                <span>Status <StatusPill status={ops?.system_status} /></span>
                <span><strong>{h.data.health_score ?? "—"}</strong>/100</span>
                <span>Human action <strong>{h.data.human_action ?? "—"}</strong></span>
                {h.data.next_expected_scan && <span>Next scan <strong>{etTime(h.data.next_expected_scan)}</strong></span>}
              </div>
              <ul className="stack-sm" aria-label="Subsystems">
                {h.data.subsystems.map((s) => <li key={s.name}><StatusPill status={s.status} /> <strong>{s.name}</strong> <span className="cap">{s.reason}</span></li>)}
              </ul>
            </Card>
            <Card title="Active incidents" id="incidents">
              {h.data.incidents.length === 0 ? <p className="cap">No incidents.</p> : (
                <ul className="stack-sm">{h.data.incidents.map((i) => (
                  <li key={i.id ?? i.summary}><StatusPill status={i.severity} /> <strong>{i.summary}</strong>
                    <div className="cap">{i.id} · {i.age_hours ?? "?"}h old{i.recommended_action ? ` · Recommended: ${i.recommended_action}` : ""}</div></li>
                ))}</ul>
              )}
            </Card>
            {ops?.workflows && ops.workflows.length > 0 && (
              <Card title="Workflows" id="workflows">
                <ul className="stack-sm">{ops.workflows.map((w) => (
                  <li key={w.workflow}><StatusPill status={w.status} /> <strong>{w.label}</strong> <span className="cap">last success {etTime(w.last_success)}</span></li>
                ))}</ul>
              </Card>
            )}
            <Card title="Email delivery" id="email" aside={h.data.email ? `Trading day ${h.data.email.day}` : undefined}>
              {!h.data.email ? <p className="cap">Email delivery status unavailable.</p> : (
                <ul className="stack-sm">{h.data.email.jobs.map((j) => (
                  <li key={j.job}><StatusPill status={j.status} /> <strong>{j.label}</strong> <span className="cap">{j.detail}</span></li>
                ))}</ul>
              )}
            </Card>
          </>
        )}
      <Card title="Services" id="services" aside={providers ? `Checked ${etTime(providers.checked_at)}` : undefined}>
        <p className="cap">Live checks of the database, Alpaca, the earnings APIs and the billing service.</p>
        <div className="row-actions">
          <button type="button" className="btn" disabled={probe.busy}
            onClick={() => void probe.run(async () => { setProviders(await unwrap(api.GET("/v1/admin/providers"))); return true; })}>
            {probe.busy ? "Checking…" : providers ? "Check again" : "Run checks"}
          </button>
        </div>
        <ErrorLine error={probe.error} />
        {providers && (
          <>
            <Checks checks={providers.checks} />
            <p className="cap">Earnings cache: {providers.earnings_cache.dated_rows ?? "?"} dated rows, last daily refresh {etTime(providers.earnings_cache.last_refresh)}.</p>
          </>
        )}
      </Card>
    </div>
  );
}

// ---- tools -------------------------------------------------------------------------------------
function EarningsRefresh() {
  const s = useApi("admin-earnings", (signal) => unwrap(api.GET("/v1/admin/earnings-refresh", { signal })));
  const act = useAction();
  const [job, setJob] = useState<Schemas["AdminEarningsRefresh"] | null>(null);
  const cur = job ?? s.data;
  const running = cur?.state === "running";
  const { reload } = s;
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => { setJob(null); reload(); }, 5000);
    return () => clearInterval(t);
  }, [running, reload]);
  return (
    <Card title="Refresh earnings" id="earnings">
      <p className="cap">Pulls the earnings calendar for the S&amp;P 500 and Nasdaq lists from FMP, then Finnhub. The daily cron does this too; use it when the calendar looks wrong.</p>
      <div className="row-actions">
        <button type="button" className="btn" disabled={running || act.busy}
          onClick={() => void act.run(async () => { setJob(await unwrap(api.POST("/v1/admin/earnings-refresh"))); return true; })}>
          {running ? "Refreshing…" : "Refresh earnings now"}
        </button>
      </div>
      <ErrorLine error={act.error} />
      {cur && cur.state !== "idle" && (
        <p className="cap" role="status">
          {cur.state === "running" ? `Running for ${cur.symbols ?? "?"} symbols, started ${etTime(cur.started_at)} by ${cur.started_by}.`
            : cur.state === "done" ? `Done ${etTime(cur.finished_at)}: dates found for ${cur.dated ?? 0} of ${cur.symbols ?? "?"} symbols.`
            : `Failed ${etTime(cur.finished_at)} (${cur.error}).`}
        </p>
      )}
    </Card>
  );
}

function DbCheck() {
  const act = useAction();
  const [out, setOut] = useState<Schemas["AdminDbCheck"] | null>(null);
  return (
    <Card title="Database checks" id="db">
      <p className="cap">Checks the database answers and the earnings table&apos;s symbols are clean (and fixes them if not).</p>
      <div className="row-actions">
        <button type="button" className="btn" disabled={act.busy} onClick={() => void act.run(async () => { setOut(await unwrap(api.POST("/v1/admin/db-check"))); return true; })}>
          {act.busy ? "Checking…" : "Run database checks"}
        </button>
      </div>
      <ErrorLine error={act.error} />
      {out && (
        <div className="stack-sm" role="status">
          <p><StatusPill status={out.connected && out.issues.length === 0 ? "ok" : out.connected ? "warn" : "error"} /> {out.connected ? "Connected" : "Not connected"} · {out.earnings_rows ?? "?"} earnings rows · {out.bad_symbols ?? "?"} symbol issues</p>
          {out.issues.length > 0 && <ul>{out.issues.map((i) => <li key={i} className="cap">{i}</li>)}</ul>}
        </div>
      )}
    </Card>
  );
}

function Activity() {
  const act = useAction();
  const [out, setOut] = useState<Schemas["AdminActivity"] | null>(null);
  const failed = out?.login_attempts.filter((a) => !a.success).length ?? 0;
  return (
    <Card title="Activity" id="activity">
      <div className="row-actions">
        <button type="button" className="btn" disabled={act.busy} onClick={() => void act.run(async () => { setOut(await unwrap(api.GET("/v1/admin/activity"))); return true; })}>
          {act.busy ? "Loading…" : out ? "Reload" : "Load activity"}
        </button>
      </div>
      <ErrorLine error={act.error} />
      {out && (
        <div className="stack">
          <h3 className="h3">Scan errors (last 50)</h3>
          {out.scan_errors.length === 0 ? <p className="cap">No scan errors recorded.</p> : (
            <div className="table-wrap"><table className="table">
              <thead><tr><th scope="col">When</th><th scope="col">Where</th><th scope="col">Error</th></tr></thead>
              <tbody>{out.scan_errors.map((e, i) => (
                <tr key={`${e.at}-${i}`}><td className="cap">{etTime(e.at)}</td><td>{e.context}{e.username ? <div className="cap">{e.username} · {e.tickers ?? "?"} tickers</div> : null}</td>
                  <td><strong>{e.error_type}</strong><div className="cap">{e.message}</div></td></tr>
              ))}</tbody>
            </table></div>
          )}
          <h3 className="h3">Sign-in attempts (last 24 hours){failed > 0 ? ` · ${failed} failed` : ""}</h3>
          {out.login_attempts.length === 0 ? <p className="cap">No sign-in attempts recorded.</p> : (
            <div className="table-wrap"><table className="table">
              <thead><tr><th scope="col">When</th><th scope="col">Account</th><th scope="col">Result</th><th scope="col" className="hide-narrow">IP</th></tr></thead>
              <tbody>{out.login_attempts.map((a, i) => (
                <tr key={`${a.at}-${i}`}><td className="cap">{etTime(a.at)}</td><td>{a.username}</td>
                  <td>{a.success ? <Pill tone="up">OK</Pill> : <Pill tone="down">Failed</Pill>}</td><td className="cap hide-narrow">{a.ip_address}</td></tr>
              ))}</tbody>
            </table></div>
          )}
          <h3 className="h3">AI feature use (30 days)</h3>
          {out.ai_usage.length === 0 ? <p className="cap">No AI use recorded.</p> : (
            <ul className="stack-sm">{out.ai_usage.map((u) => <li key={u.feature}><strong>{u.calls}</strong> <span className="cap">{u.feature}</span></li>)}</ul>
          )}
        </div>
      )}
    </Card>
  );
}

function ToolsTab() {
  return (
    <div className="stack">
      <EarningsRefresh />
      <DbCheck />
      <Activity />
    </div>
  );
}

export function AdminView() {
  const { me } = useSession();
  const [tab, setTab] = useState<Tab>("users");
  if (!me?.is_admin) {
    return <Card><Empty title="Admins only.">This page is for HSF administrators.</Empty></Card>;
  }
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Admin</h1>
          <p className="cap">Accounts, system health and maintenance tools. Account changes are recorded in the admin log.</p>
        </div>
      </section>
      <div className="seg-group" role="tablist" aria-label="Admin sections">
        {TABS.map(([k, label]) => (
          <button key={k} type="button" role="tab" aria-selected={tab === k} className={`seg${tab === k ? " on" : ""}`} onClick={() => setTab(k)}>{label}</button>
        ))}
      </div>
      {tab === "users" ? <UsersTab self={me.email ?? ""} /> : tab === "health" ? <HealthTab /> : <ToolsTab />}
    </div>
  );
}
