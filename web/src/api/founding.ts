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
