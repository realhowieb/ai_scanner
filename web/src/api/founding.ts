// Founding 100 (signed in): the offer, a one-time founder checkout and giving back an
// unfinished one. Prices and seats always come from the API.
import { api, unwrap } from "./client";
import type { Schemas } from "./client";

export type FoundingOffer = Schemas["FoundingOffer"];
export type FounderTier = Schemas["OfferTier"]["tier"];

export const founding = {
  offer: (signal?: AbortSignal) => unwrap(api.GET("/v1/founding/offer", { signal })),
  checkout: (offer: FoundingOffer, tier: FounderTier, acknowledged: boolean, upgrade = false) =>
    unwrap(api.POST("/v1/founding/checkout", {
      body: {
        tier, upgrade, acknowledged,
        policies: offer.policies.map((p) => ({ policy_type: p.policy_type, version: p.version, content_hash: p.content_hash })),
      },
    })),
  cancel: () => unwrap(api.POST("/v1/founding/checkout/cancel")),
};

// Refunds: a member asks about their own payments (the API decides which, from the
// session); admins review one request at a time.
export type MyRefunds = Schemas["MyRefunds"];
export type FounderPayment = Schemas["FounderPayment"];
export type RefundRequest = Schemas["RefundRequest"];
export type AdminRefundRequest = Schemas["AdminRefundRequest"];
export type RefundBasis = Schemas["ApproveBody"]["basis"];
export type RefundStatus = Schemas["RefundRequest"]["status"];

export const refunds = {
  mine: (signal?: AbortSignal) => unwrap(api.GET("/v1/founding/refunds/me", { signal })),
  request: (kind: FounderPayment["kind"], reason: string) =>
    unwrap(api.POST("/v1/founding/refunds/request", { body: { kind, reason } })),
  list: (status?: RefundStatus, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/admin/founding/refunds", { params: { query: status ? { status } : {} }, signal })),
  detail: (id: number, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/admin/founding/refunds/{request_id}", { params: { path: { request_id: id } }, signal })),
  prorated: (id: number, stoppedAt: string) =>
    unwrap(api.GET("/v1/admin/founding/refunds/{request_id}/prorated",
      { params: { path: { request_id: id }, query: { stopped_at: stoppedAt } } })),
  approve: (id: number, body: Schemas["ApproveBody"]) =>
    unwrap(api.POST("/v1/admin/founding/refunds/{request_id}/approve", { params: { path: { request_id: id } }, body })),
  deny: (id: number, reason: string) =>
    unwrap(api.POST("/v1/admin/founding/refunds/{request_id}/deny", { params: { path: { request_id: id } }, body: { reason } })),
};

export const cents = (n: number | null | undefined) =>
  `$${((n ?? 0) / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const REFUND_STATUS_LABEL: Record<RefundStatus, string> = {
  requested: "Waiting for review", approved: "Approved", denied: "Not approved", initiated: "Refund started",
  completed: "Refunded", failed: "Refund failed",
};
