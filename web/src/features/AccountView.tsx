"use client";

// Account: identity, plan and what it includes (from /v1/me only), billing through the
// existing Stripe flow (checkout / customer portal), email settings, password, sign
// out and account deletion. Every change is shown only after the server confirms it.
import Link from "next/link";
import { useState } from "react";
import type { FormEvent } from "react";

import { account } from "@/api/account";
import type { EmailPrefs } from "@/api/account";
import { Dialog } from "@/components/Dialog";
import { Card, ErrorLine, ErrorState, Pill, Skeleton, UpgradeButton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { FEATURE_PLAN, signOut, useSession } from "@/session/SessionProvider";
import type { Me } from "@/session/SessionProvider";

/** What each entitlement means to a customer. Unknown flags are not shown. */
export const FEATURE_LABELS: Record<string, string> = {
  can_scan_sp500: "S&P 500 scans",
  can_scan_nasdaq: "NASDAQ and Combo scans",
  can_full_universe: "Full US market scans",
  can_premarket: "Pre-market scans and movers",
  can_afterhours: "After-hours scans and movers",
  can_unusual_volume: "Unusual volume filter",
  can_early_breakout: "PreBreakout probability",
  can_scan_history: "Scan history",
  can_track_record: "Historical research (track record)",
  can_earnings: "Earnings calendar",
  can_day_trader: "Day Trader",
  can_email_alerts: "Alert emails",
  can_ai_notes: "AI notes and summaries",
  can_paper_trade: "Paper trading",
  can_export_csv: "CSV export",
};

const PREFS: { key: keyof EmailPrefs; label: string; note: string }[] = [
  { key: "digest", label: "Morning market digest", note: "Before the open" },
  { key: "evening", label: "Evening market wrap", note: "After the close" },
  { key: "alerts", label: "Alert emails", note: "In-app alerts continue either way" },
];

function PlanCard({ me }: { me: Me }) {
  const portal = useAction();
  const open = (flow?: "cancel") => void portal.run(async () => {
    const link = await account.portal(flow);
    window.location.assign(link.url);
    return true;
  });
  const included = Object.keys(FEATURE_LABELS).filter((k) => me.entitlements[k]);
  const locked = Object.keys(FEATURE_LABELS).filter((k) => k in me.entitlements && !me.entitlements[k]);
  const paid = me.plan === "pro" || me.plan === "premium";
  return (
    <Card title="Plan" id="plan" aside={<Pill tone={paid ? "gold" : undefined}>{me.plan_label}</Pill>}>
      {me.plan === "admin" ? (
        <p className="body-sm">Admin account. Every feature is on and billing doesn&apos;t apply.</p>
      ) : (
        <div className="row-actions">
          {me.plan === "basic" && <UpgradeButton plan="pro" />}
          {me.plan !== "premium" && <UpgradeButton plan="premium" />}
          {paid && (
            <button type="button" className="btn" disabled={portal.busy} onClick={() => open()}>
              {portal.busy ? "Opening…" : "Manage subscription"}
            </button>
          )}
        </div>
      )}
      <ErrorLine error={portal.error} />
      {!me.email_verified && me.plan !== "admin" && <p className="cap">Verify your email before upgrading.</p>}
      {paid && <p className="cap">Change plan, payment method or invoices in the Stripe customer portal. <button type="button" className="link-btn" onClick={() => open("cancel")}>Cancel subscription</button></p>}
      <div className="grid2">
        <div className="stack-sm">
          <p className="strong">Included</p>
          <ul className="bullets">{included.map((k) => <li key={k}>{FEATURE_LABELS[k]}</li>)}</ul>
          <p className="cap">Up to {me.alert_limit} alert{me.alert_limit === 1 ? "" : "s"}.</p>
        </div>
        {locked.length > 0 && (
          <div className="stack-sm">
            <p className="strong">Not on your plan</p>
            <ul className="bullets muted-list">
              {locked.map((k) => (
                <li key={k}>{FEATURE_LABELS[k]}{k in FEATURE_PLAN ? <span className="cap"> · {FEATURE_PLAN[k as keyof typeof FEATURE_PLAN] === "pro" ? "Pro" : "Premium"}</span> : null}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}

function ProfileCard({ me }: { me: Me }) {
  const resend = useAction();
  const [sent, setSent] = useState<string | null>(null);
  return (
    <Card title="Profile" id="profile">
      <dl className="kv">
        {me.name && <div><dt>Name</dt><dd>{me.name}</dd></div>}
        <div><dt>Email</dt><dd className="wrap">{me.email}</dd></div>
        <div><dt>Email status</dt><dd>{me.email_verified ? <Pill tone="up">Verified</Pill> : <Pill tone="warn">Not verified</Pill>}</dd></div>
      </dl>
      {!me.email_verified && (
        <div className="stack-sm">
          <p className="cap">Verifying is needed to upgrade and for email alerts.</p>
          <div className="row-actions">
            <button type="button" className="btn" disabled={resend.busy}
              onClick={() => void resend.run(async () => { setSent((await account.resendVerification()).message); return true; })}>
              {resend.busy ? "Sending…" : "Send verification email"}
            </button>
          </div>
          {sent && <p className="notice" role="status">{sent}</p>}
          <ErrorLine error={resend.error} />
        </div>
      )}
    </Card>
  );
}

function EmailCard({ me }: { me: Me }) {
  const prefs = useApi("email-prefs", (signal) => account.emailPrefs(signal));
  const [saved, setSaved] = useState<EmailPrefs | null>(null);
  const act = useAction();
  const current = saved ?? prefs.data;
  const toggle = (key: keyof EmailPrefs, on: boolean) => void act.run(async () => {
    setSaved(await account.setEmailPrefs({ [key]: on }));
    return true;
  });
  return (
    <Card title="Email" id="email">
      {prefs.error && !current ? <ErrorState error={prefs.error} onRetry={prefs.reload} what="your email settings" /> :
        !current ? <Skeleton rows={3} label="Loading email settings" /> : (
          <>
            <div className="stack-xs" role="group" aria-label="Emails HSF sends you">
              {PREFS.map((p) => (
                <label key={p.key} className="check">
                  <input type="checkbox" checked={current[p.key]} disabled={act.busy} onChange={(e) => toggle(p.key, e.target.checked)} />
                  <span className="grow">{p.label}</span><span className="cap">{p.note}</span>
                </label>
              ))}
            </div>
            <ErrorLine error={act.error} />
            <p className="cap">
              {me.entitlements.can_email_alerts ? "" : "Market emails and alert emails go to Pro and Premium accounts. "}
              Account emails (verification, password reset) always go out.
            </p>
          </>
        )}
    </Card>
  );
}

function PasswordCard() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [done, setDone] = useState(false);
  const act = useAction();
  const mismatch = !!again && next !== again;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!current || !next || next !== again) return;
    setDone(false);
    const ok = await act.run(async () => { await account.changePassword(current, next); return true; });
    if (ok) {
      setCurrent(""); setNext(""); setAgain("");
      setDone(true);
    }
  };
  return (
    <Card title="Password" id="password">
      <form className="stack-sm" onSubmit={submit}>
        <label className="field"><span>Current password</span>
          <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} required />
        </label>
        <label className="field"><span>New password</span>
          <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} required />
        </label>
        <label className="field"><span>Repeat new password</span>
          <input type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} required
            aria-invalid={mismatch || undefined} aria-describedby={mismatch ? "pw-mismatch" : undefined} />
        </label>
        {mismatch && <p id="pw-mismatch" className="form-error">The new passwords don&apos;t match.</p>}
        <ErrorLine error={act.error} />
        {done && <p className="notice" role="status">Password changed. Other devices and the classic app were signed out.</p>}
        <div className="row-actions">
          <button type="submit" className="btn btn-primary" disabled={act.busy || !current || !next || next !== again}>
            {act.busy ? "Changing…" : "Change password"}
          </button>
        </div>
      </form>
    </Card>
  );
}

/** Hands the browser a file to save; the BFF doesn't pass Content-Disposition through. */
export function saveJson(data: unknown, filename: string): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function DownloadData() {
  const act = useAction();
  const [partial, setPartial] = useState<string[] | null>(null);
  const download = async () => {
    setPartial(null);
    const data = await act.run(() => account.exportData());
    if (!data) return;
    saveJson(data, `hsf-data-${new Date().toISOString().slice(0, 10)}.json`);
    const missing = (data as { unavailable?: string[] }).unavailable ?? [];
    setPartial(missing.length ? missing : []);
  };
  return (
    <Card title="Your data" id="data">
      <p className="body-sm">Download your watchlists, alerts, journal, saved scans and settings as one JSON file.</p>
      <ErrorLine error={act.error} />
      {partial && partial.length > 0 && (
        <p className="notice" role="status">Downloaded, but these couldn&apos;t be read right now: {partial.join(", ").replaceAll("_", " ")}. Try again in a minute for a complete copy.</p>
      )}
      {partial && partial.length === 0 && <p className="notice" role="status">Downloaded.</p>}
      <div className="row-actions">
        <button type="button" className="btn" onClick={() => void download()} disabled={act.busy}>{act.busy ? "Preparing…" : "Download my data"}</button>
      </div>
    </Card>
  );
}

function DeleteAccount() {
  const [open, setOpen] = useState(false);
  const [password, setPassword] = useState("");
  const [typed, setTyped] = useState("");
  const act = useAction();
  const close = () => { setOpen(false); setPassword(""); setTyped(""); act.clear(); };
  const confirm = async (e: FormEvent) => {
    e.preventDefault();
    if (!password || typed !== "DELETE") return;
    const ok = await act.run(async () => { await account.remove(password); return true; });
    if (ok) await signOut();
  };
  return (
    <Card title="Delete account" id="delete">
      <p className="body-sm">Permanently deletes your account with its watchlists, alerts, journal, saved scans and settings. Cancel a paid subscription first.</p>
      <div className="row-actions"><button type="button" className="btn btn-danger-outline" onClick={() => setOpen(true)}>Delete account…</button></div>
      <Dialog open={open} title="Delete your account?" onClose={close}>
        <form className="stack-sm" onSubmit={confirm}>
          <p className="body-sm">This can&apos;t be undone.</p>
          <label className="field"><span>Password</span>
            <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} data-autofocus />
          </label>
          <label className="field"><span>Type DELETE to confirm</span>
            <input value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" />
          </label>
          <ErrorLine error={act.error} />
          <div className="row-actions end">
            <button type="button" className="btn" onClick={close}>Keep my account</button>
            <button type="submit" className="btn btn-danger" disabled={act.busy || !password || typed !== "DELETE"}>{act.busy ? "Deleting…" : "Delete account"}</button>
          </div>
        </form>
      </Dialog>
    </Card>
  );
}

export function AccountView() {
  const { me } = useSession();
  if (!me) return null;
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Account</h1>
          <p className="cap">Your profile, plan, billing and email settings.</p>
        </div>
        <button type="button" className="btn" onClick={() => void signOut()}>Sign out</button>
      </section>
      <div className="split">
        <div className="col-main">
          <PlanCard me={me} />
          <EmailCard me={me} />
          <p className="cap">Alerts and watchlists have their own pages: <Link href="/alerts">Alerts</Link> · <Link href="/watchlists">Watchlists</Link>.</p>
        </div>
        <div className="col-side">
          <ProfileCard me={me} />
          <PasswordCard />
          <DownloadData />
          <DeleteAccount />
        </div>
      </div>
    </div>
  );
}
