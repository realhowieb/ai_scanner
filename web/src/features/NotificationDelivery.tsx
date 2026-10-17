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
  const [saved, setSaved] = useState(false);
  const act = useAction();
  return <form className="stack-sm" onSubmit={(e) => { e.preventDefault(); void act.run(async () => {
    await saveWorkspace({ quiet_hours: hours }); setSaved(true); return true;
  }); }}>
    <label><input type="checkbox" checked={hours.enabled ?? false} onChange={(e) => { setHours({ ...hours, enabled: e.target.checked }); setSaved(false); }} /> Enable quiet hours</label>
    <div className="row-actions">
      <label className="field">From<input type="time" required value={hours.start ?? "22:00"} onChange={(e) => setHours({ ...hours, start: e.target.value })} /></label>
      <label className="field">Until<input type="time" required value={hours.end ?? "07:00"} onChange={(e) => setHours({ ...hours, end: e.target.value })} /></label>
    </div>
    <label className="field">Time zone<input required value={hours.timezone ?? "America/New_York"} placeholder="America/New_York" maxLength={64} onChange={(e) => setHours({ ...hours, timezone: e.target.value })} /></label>
    <p className="cap">Suppresses browser pushes and alert emails during this interval; alerts remain in Recently fired. Skipped messages are not sent later. Equal start and end times disable the interval.</p>
    <div><button className="btn btn-sm" disabled={act.busy}>Save quiet hours</button></div>
    {saved && <p className="cap" role="status">Quiet hours saved.</p>}
    <ErrorLine error={act.error} />
  </form>;
}
export function NotificationDelivery({ events = [] }: { events?: Schemas["AlertEvent"][] }) {
  const prefs = useWorkspace();
  const deliveries = useApi("push-deliveries", (signal) => unwrap(api.GET("/v1/alerts/deliveries", { signal })));
  const act = useAction();
  return <>
    <Card title="Quiet hours" id="quiet-hours">
      {prefs.data && <QuietHoursForm key={JSON.stringify(prefs.data.quiet_hours)} initial={prefs.data.quiet_hours ?? { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" }} />}
      <ErrorLine error={prefs.error} />
    </Card>
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
  </>;
}
