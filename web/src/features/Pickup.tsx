// When HSF picked a stock up: the first scan of its current unbroken run of full-market
// scans (API `pickup`), with the price then and the move since. Exact to that scan (scans
// run about hourly in market hours), not to the minute.
import type { Schemas } from "@/api/client";
import { pct, price } from "@/lib/format";

export type PickupInfo = Schemas["Pickup"];

const ET = "America/New_York";
const dayKey = (d: Date) => d.toLocaleDateString("en-US", { timeZone: ET, year: "numeric", month: "2-digit", day: "2-digit" });
const upDown = (v: number | null | undefined) => (v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "";

/** "9:35 AM ET" today (ET), "Mon 12:35 PM ET" within the week, "Nov 3, 12:35 PM ET" before that. */
export function pickupWhen(at: string | null | undefined, nowMs: number = Date.now()): string {
  if (!at) return "";
  const d = new Date(at);
  if (Number.isNaN(d.getTime())) return "";
  const time = d.toLocaleTimeString("en-US", { timeZone: ET, hour: "numeric", minute: "2-digit" });
  const now = new Date(nowMs);
  if (dayKey(d) === dayKey(now)) return `${time} ET`;
  const ageDays = (nowMs - d.getTime()) / 86_400_000;
  const day = ageDays < 6
    ? d.toLocaleDateString("en-US", { timeZone: ET, weekday: "short" })
    : `${d.toLocaleDateString("en-US", { timeZone: ET, month: "short", day: "numeric" })},`;
  return `${day} ${time} ET`;
}

/** Price now vs the price when HSF picked it up, in %. Null without both prices. */
export function sincePickup(p: PickupInfo | null | undefined, last: number | null | undefined): number | null {
  if (!p || p.price == null || last == null || !(p.price > 0)) return null;
  return (last / p.price - 1) * 100;
}

function explain(p: PickupInfo, nowMs?: number): string {
  const when = pickupWhen(p.at, nowMs);
  const head = p.open_ended ? `In every HSF scan since at least the ${when} scan` : `First picked up by HSF in the ${when} scan`;
  const rank = p.rank != null ? `, ranked #${p.rank}` : "";
  return `${head}${p.price != null ? ` at ${price(p.price)}` : ""}${rank}. Exact to that scan; HSF scans the market about hourly in market hours.`;
}

/** Scanner table cell / phone card fact: the time, then the price then and the move since. */
export function PickedCell({ pickup, last, nowMs }: { pickup: PickupInfo | null | undefined; last: number | null | undefined; nowMs?: number }) {
  if (!pickup) return <span className="cap">—</span>;
  const move = sincePickup(pickup, last);
  return <span className="picked-cell" title={explain(pickup, nowMs)}>
    <span className="mono">{pickup.open_ended ? "≤ " : ""}{pickupWhen(pickup.at, nowMs)}</span>
    {pickup.price != null && <span className="cap mono">{price(pickup.price)}{move != null && Math.abs(move) >= 0.05 && <> · <span className={upDown(move)}>{pct(move, 1)}</span></>}</span>}
  </span>;
}

/** One sentence for the quick view and the stock page. */
export function PickupLine({ pickup, last, nowMs }: { pickup: PickupInfo | null | undefined; last: number | null | undefined; nowMs?: number }) {
  if (!pickup) return null;
  const when = pickupWhen(pickup.at, nowMs);
  const move = sincePickup(pickup, last);
  return <p className="cap pickup-line" title="Exact to that scan; HSF scans the market about hourly in market hours.">
    {pickup.open_ended ? <>In every HSF scan since at least <strong>{when}</strong></> : <>Picked up by HSF at <strong>{when}</strong></>}
    {pickup.price != null && <> at <span className="mono">{price(pickup.price)}</span></>}
    {pickup.rank != null && <> (#{pickup.rank} then)</>}
    {move != null && Math.abs(move) >= 0.05 && <> · <span className={`mono ${upDown(move)}`}>{pct(move, 1)}</span> since</>}
  </p>;
}
