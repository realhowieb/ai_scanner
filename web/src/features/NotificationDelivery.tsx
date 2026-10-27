"use client";

import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, Empty, ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { saveWorkspace, useWorkspace } from "@/hooks/useWorkspace";
import { etTime } from "@/lib/format";

function QuietHoursForm({ initial }: { initial: Schemas["QuietHours"] }) {
  const [hours, setHours] = useState(initial);
  const zones = [...new Set([initial.timezone ?? "America/New_York", "UTC", ...(typeof Intl.supportedValuesOf === "function" ? Intl.supportedValuesOf("timeZone") : ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"])])];
  const [saved, setSaved] = useState(false);
  const act = useAction();
  return <form className="stack-sm" onSubmit={(e) => { e.preventDefault(); void act.run(async () => {
    await saveWorkspace({ quiet_hours: hours }); setSaved(true); return true;
  }); }}>
    <label><input type="checkbox" checked={hours.enabled ?? false} onChange={(e) => { setHours({ ...hours, enabled: e.target.checked }); setSaved(false); }} /> Enable quiet hours</label>
    <div className="row-actions">
      <label className="field">From<input type="time" required value={hours.start ?? "22:00"} onChange={(e) => { setHours({ ...hours, start: e.target.value }); setSaved(false); }} /></label>
      <label className="field">Until<input type="time" required value={hours.end ?? "07:00"} onChange={(e) => { setHours({ ...hours, end: e.target.value }); setSaved(false); }} /></label>
    </div>
    <label className="field">Time zone<select required value={hours.timezone ?? "America/New_York"} onChange={(e) => { setHours({ ...hours, timezone: e.target.value }); setSaved(false); }}>{zones.map((zone) => <option key={zone} value={zone}>{zone.replaceAll("_", " ")}</option>)}</select></label>
    <p className="cap">Suppresses browser pushes and alert emails during this interval; alerts remain in Recently fired. Skipped messages are not sent later. Equal start and end times disable the interval.</p>
    <div><button className="btn btn-sm" disabled={act.busy}>Save quiet hours</button></div>
    {saved && <p className="cap" role="status">Quiet hours saved.</p>}
    <ErrorLine error={act.error} />
  </form>;
}
/** Telegram and Discord: every alert that fires is also posted there (Pro and above). */
function ChatDelivery() {
  const chats = useApi("chat-alerts", (signal) => unwrap(api.GET("/v1/me/chat-alerts", { signal })));
  const act = useAction();
  const [webhook, setWebhook] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const d = chats.data;
  if (!d?.telegram || !d?.discord) return null; // not loaded, or an API without chat alerts
  if (!d.telegram.available && !d.discord.available) return null;
  const status = (c: Schemas["ChatDestination"]) => c.last_status ? ` · last alert ${c.last_status}${c.last_at ? ` ${etTime(c.last_at)}` : ""}` : "";
  return (
    <Card title="Telegram and Discord" id="chat-delivery">
      <p className="cap">Every alert that fires is also posted to the chats you connect here, except during quiet hours.</p>
      {!d.plan_allows ? <p className="cap">Telegram and Discord alerts are part of <a href="/pricing">Pro</a>.</p> : <div className="stack-sm">
        {d.telegram.available && <div className="chat-row">
          <strong>Telegram</strong>
          {d.telegram.connected ? <>
            <span className="cap">Connected to {d.telegram.label ?? "a chat"}{status(d.telegram)}</span>
            <button type="button" className="btn btn-sm" disabled={act.busy} onClick={() => void act.run(async () => {
              await unwrap(api.DELETE("/v1/me/chat-alerts/{kind}", { params: { path: { kind: "telegram" } } })); chats.reload(); return true;
            })}>Disconnect</button>
          </> : <button type="button" className="btn btn-sm" disabled={act.busy} onClick={() => void act.run(async () => {
            const link = await unwrap(api.POST("/v1/me/chat-alerts/telegram", {}));
            window.open(link.url, "_blank", "noopener,noreferrer");
            setNote(`Telegram opened. Tap Start in the bot chat within ${link.expires_in_minutes} minutes, then refresh this page.`);
            return true;
          })}>Connect Telegram</button>}
        </div>}
        {d.discord.available && <div className="stack-sm">
          <div className="chat-row">
            <strong>Discord</strong>
            {d.discord.connected && <>
              <span className="cap">Connected to a channel webhook{status(d.discord)}</span>
              <button type="button" className="btn btn-sm" disabled={act.busy} onClick={() => void act.run(async () => {
                await unwrap(api.DELETE("/v1/me/chat-alerts/{kind}", { params: { path: { kind: "discord" } } })); chats.reload(); return true;
              })}>Disconnect</button>
            </>}
          </div>
          <form className="chat-row" onSubmit={(e) => { e.preventDefault(); void act.run(async () => {
            await unwrap(api.PUT("/v1/me/chat-alerts/discord", { body: { webhook_url: webhook.trim() } }));
            setWebhook(""); setNote("Discord connected. A test message was posted in the channel."); chats.reload(); return true;
          }); }}>
            <label className="field grow"><span>{d.discord.connected ? "Replace webhook URL" : "Discord channel webhook URL"}</span>
              <input type="url" inputMode="url" autoComplete="off" placeholder="https://discord.com/api/webhooks/…" value={webhook} onChange={(e) => setWebhook(e.target.value)} />
            </label>
            <button className="btn btn-sm" disabled={act.busy || webhook.trim().length < 30}>Connect Discord</button>
          </form>
          <p className="cap">In Discord: Channel settings → Integrations → Webhooks → New Webhook → Copy Webhook URL.</p>
        </div>}
        {(d.telegram.connected || d.discord.connected) && <div><button type="button" className="btn btn-sm" disabled={act.busy} onClick={() => void act.run(async () => {
          const r = await unwrap(api.POST("/v1/me/chat-alerts/test", {})); setNote(r.message); chats.reload(); return true;
        })}>Send a test alert</button></div>}
      </div>}
      {note && <p className="cap" role="status">{note}</p>}
      <ErrorLine error={act.error} />
    </Card>
  );
}

export function NotificationDelivery({ events = [] }: { events?: Schemas["AlertEvent"][] }) {
  const prefs = useWorkspace();
  const deliveries = useApi("push-deliveries", (signal) => unwrap(api.GET("/v1/alerts/deliveries", { signal })));
  const act = useAction();
  return <>
    <ChatDelivery />
    <Card title="Quiet hours" id="quiet-hours">
      {prefs.data && <QuietHoursForm key={JSON.stringify(prefs.data.quiet_hours)} initial={prefs.data.quiet_hours ?? { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" }} />}
      <ErrorLine error={prefs.error} />
    </Card>
    <details className="delivery-diagnostics"><summary>Delivery diagnostics</summary><div className="stack-sm">
    <Card title="Alert email delivery" id="email-delivery">
      <p className="cap">Recent rule email statuses. Failed rule emails are retried automatically by the alert worker.</p>
      <ul className="stack-sm">{events.filter((e) => e.delivery?.email).map((e) => <li key={e.event_id}>
        <strong>{e.message}</strong><p className="cap">{e.delivery.email} · {etTime(e.fired_at)}</p>
      </li>)}</ul>
      {!events.some((e) => e.delivery?.email) && <p className="cap">No recent rule email delivery statuses.</p>}
    </Card>
    <Card title="Browser notification delivery" id="notification-delivery">
      <p className="cap">Accepted means the browser push service accepted the message; it does not confirm it was displayed or read. Recent attempts for each browser are shown separately. Unconfirmed attempts cannot be retried because the service may already have received them.</p>
      <div><button type="button" className="btn btn-sm" onClick={deliveries.reload}>Refresh delivery</button></div>
      <ErrorLine error={deliveries.error ?? act.error} />
      {deliveries.data?.length === 0 && <Empty title="No browser delivery attempts recorded yet." />}
      <ul className="stack-sm">{deliveries.data?.map((d) => <li key={d.id}>
        <strong>{d.title}</strong><p className="cap">{d.status.replaceAll("_", " ")} · {d.attempts} attempt{d.attempts === 1 ? "" : "s"} · {etTime(d.at)}</p>
        {d.retryable && <button type="button" className="btn btn-sm" disabled={act.busy} onClick={() => void act.run(async () => {
          await unwrap(api.POST("/v1/alerts/deliveries/{delivery_id}/retry", { params: { path: { delivery_id: d.id } } }));
          deliveries.reload(); return true;
        })}>Retry notification</button>}
      </li>)}</ul>
    </Card>
    </div></details>
  </>;
}
