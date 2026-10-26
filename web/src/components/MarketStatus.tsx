"use client";

import { useEffect } from "react";

import { api, unwrap } from "@/api/client";
import { useApi } from "@/hooks/useApi";

type Phase = "premarket" | "open" | "afterhours" | "closed";

export const MARKET_PHASES: Record<Phase, { label: string; tone: "up" | "amber" | "muted" }> = {
  open: { label: "Market open", tone: "up" },
  premarket: { label: "Pre-market", tone: "amber" },
  afterhours: { label: "After hours", tone: "amber" },
  closed: { label: "Market closed", tone: "muted" },
};

const RECHECK_MS = 5 * 60_000;

/** The U.S. session as the API reports it (holiday-aware), in the header on every signed-in page.
 * Shares the price strip's request. Hidden when the API doesn't say, rather than guessed from the clock. */
export function MarketStatus() {
  const { data, reload } = useApi("tape", (signal) => unwrap(api.GET("/v1/market/tape", { signal })));
  useEffect(() => {
    const t = window.setInterval(() => { if (document.visibilityState === "visible") reload(); }, RECHECK_MS);
    return () => window.clearInterval(t);
  }, [reload]);
  const phase = data?.phase ? MARKET_PHASES[data.phase as Phase] : undefined;
  if (!phase) return null;
  return (
    <span className={`market-status tone-${phase.tone}`} title="U.S. stock market session, Eastern Time. HSF prices come from scheduled scans and delayed quotes, not a live feed.">
      <span className="dot" aria-hidden="true" />
      {phase.label}
    </span>
  );
}
