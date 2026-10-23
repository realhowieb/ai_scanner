"use client";

import { useState } from "react";

import { insights } from "@/api/userData";
import { captureUsage } from "@/lib/usageAnalytics";

/** The public link for a stock (/share/X), carrying your invite code so a friend who signs up
 * from it is credited to you. Uses the phone's share sheet when there is one, else copies. */
export function shareUrl(origin: string, ticker: string, code: string | null | undefined): string {
  const url = new URL(`/share/${encodeURIComponent(ticker)}`, origin);
  if (code && /^[a-z0-9]{4,32}$/.test(code)) url.searchParams.set("ref", code);
  return url.toString();
}

export function ShareStockButton({ ticker }: { ticker: string }) {
  const [note, setNote] = useState<string | null>(null);
  const share = async () => {
    let code: string | null = null;
    try { code = (await insights.referrals()).code; } catch { /* share without credit */ }
    const url = shareUrl(window.location.origin, ticker, code);
    const nav = navigator as Navigator & { share?: (d: ShareData) => Promise<void> };
    if (typeof nav.share === "function" && window.matchMedia?.("(pointer: coarse)").matches) {
      try {
        await nav.share({ title: `${ticker} on HSF`, text: `${ticker} on HSF`, url });
        captureUsage("stock_shared", { method: "native" });
      } catch { /* closed the share sheet */ }
      return;
    }
    try {
      await navigator.clipboard.writeText(url);
      captureUsage("stock_shared", { method: "copy" });
      setNote("Link copied");
    } catch {
      setNote(url);
    }
    window.setTimeout(() => setNote(null), 4000);
  };
  return (
    <>
      <button type="button" className="btn share-btn" onClick={() => void share()}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7M16 6l-4-4-4 4M12 2v14" /></svg>
        Share
      </button>
      {note && <span className="cap" role="status">{note}</span>}
    </>
  );
}
