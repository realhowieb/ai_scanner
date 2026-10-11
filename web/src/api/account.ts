// Account, billing and historical-research calls. Password changes go through this
// site's /api/auth/password (the API answers with tokens, which stay in cookies).
import { ApiError, api, newRequestId, toApiError, unwrap } from "./client";
import type { Schemas } from "./client";

export type EmailPrefs = Schemas["EmailPrefs"];
export type TrackRecord = Schemas["TrackRecord"];
export type TrackRecordSummary = Schemas["TrackRecordSummary"];

export const account = {
  emailPrefs: (signal?: AbortSignal) => unwrap(api.GET("/v1/me/email-preferences", { signal })),
  setEmailPrefs: (patch: Partial<EmailPrefs>) => unwrap(api.PATCH("/v1/me/email-preferences", { body: patch })),
  resendVerification: () => unwrap(api.POST("/v1/me/verify-email")),
  portal: (flow?: "cancel") => unwrap(api.POST("/v1/billing/portal", { body: flow ? { flow } : {} })),
  checkout: (plan: "pro" | "premium", interval: "month" | "year" = "month") =>
    unwrap(api.POST("/v1/billing/checkout", { body: { plan, interval } })),
  exportData: () => unwrap(api.GET("/v1/me/export")),
  remove: (password: string) => unwrap(api.DELETE("/v1/me", { body: { password, confirm: "DELETE" } })),
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    let res: Response;
    try {
      res = await fetch("/api/auth/password", {
        method: "POST",
        credentials: "same-origin",
        headers: { "content-type": "application/json", "x-request-id": newRequestId() },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      });
    } catch {
      throw new ApiError(0, "You appear to be offline, or the site couldn't be reached.", null, null, null);
    }
    if (!res.ok) {
      let body: unknown = null;
      try {
        body = await res.json();
      } catch {
        /* keep null */
      }
      throw toApiError(res, body);
    }
  },
};

export const research = {
  trackRecord: (signal?: AbortSignal) => unwrap(api.GET("/v1/track-record", { signal })),
  daily: (ranking: "breakout" | "prebreakout", horizon: number, days = 120, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/track-record/daily", { params: { query: { ranking, horizon, days } }, signal })),
};

// Outcome Intelligence (/v1/outcomes/*, Pro). Every call sends only the filters it
// names; defaults are the complete dataset.
export type OutcomeSummary = Schemas["OutcomeSummary"];
export type OutcomeScores = Schemas["OutcomeScores"];
export type OutcomeHorizons = Schemas["OutcomeHorizons"];
export type OutcomeGroups = Schemas["OutcomeGroups"];
export type OutcomeSymbol = Schemas["OutcomeSymbol"];
export type OutcomeMetrics = Schemas["OutcomeMetrics"];

export const outcomes = {
  summary: (signal?: AbortSignal) => unwrap(api.GET("/v1/outcomes/summary", { signal })),
  scores: (signal?: AbortSignal) => unwrap(api.GET("/v1/outcomes/scores", { signal })),
  horizons: (signal?: AbortSignal) => unwrap(api.GET("/v1/outcomes/horizons", { signal })),
  setups: (signal?: AbortSignal) => unwrap(api.GET("/v1/outcomes/setups", { signal })),
  symbol: (ticker: string, signal?: AbortSignal) =>
    unwrap(api.GET("/v1/outcomes/symbols/{ticker}", { params: { path: { ticker }, query: { page_size: 1 } }, signal })),
};
