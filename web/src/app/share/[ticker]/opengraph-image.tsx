import { ImageResponse } from "next/og";

import { setupLabel } from "@/lib/format";
import { loadPublicStock } from "@/server/publicStock";

export const alt = "HSF stock snapshot";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const scoreColor = (s: number) => (s >= 75 ? "#3dd68c" : s >= 55 ? "#4cc2ff" : "#f5b34b");

/** The preview card for a shared stock link: ticker, price move and, when ranked, the HSF Score. */
export default async function Image({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  const s = await loadPublicStock(ticker);
  const t = s?.ticker ?? decodeURIComponent(ticker).toUpperCase().slice(0, 10);
  const chg = s?.change_pct;
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between",
        background: "#0b0f14", color: "#e7edf5", padding: "60px 72px", fontFamily: "sans-serif" }}>
        <div style={{ display: "flex", fontSize: 34, fontWeight: 700 }}><span>HSFinest</span><span style={{ color: "#4cc2ff" }}>.AI</span></div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", fontSize: 110, fontWeight: 700, letterSpacing: -2 }}>{t}</div>
            {s?.name && <div style={{ display: "flex", fontSize: 30, color: "#b9c6d6", maxWidth: 640 }}>{s.name.slice(0, 48)}</div>}
            {s?.price != null && (
              <div style={{ display: "flex", gap: 20, fontSize: 40, marginTop: 18 }}>
                <span>{`$${s.price.toFixed(2)}`}</span>
                {chg != null && <span style={{ color: chg >= 0 ? "#3dd68c" : "#ff7a7a" }}>{`${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`}</span>}
              </div>
            )}
          </div>
          {s?.ranked && s.hsf_score != null ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", background: "#121821", border: "2px solid #243042",
              borderRadius: 24, padding: "28px 40px" }}>
              <div style={{ display: "flex", fontSize: 26, color: "#9aa8ba" }}>HSF Score</div>
              <div style={{ display: "flex", fontSize: 120, fontWeight: 700, color: scoreColor(s.hsf_score), lineHeight: 1 }}>{s.hsf_score}</div>
              {s.setup && <div style={{ display: "flex", fontSize: 26, color: "#b9c6d6", marginTop: 8 }}>{setupLabel(s.setup)}</div>}
            </div>
          ) : (
            <div style={{ display: "flex", fontSize: 28, color: "#9aa8ba", maxWidth: 380 }}>See how HSF scores it, free.</div>
          )}
        </div>
        <div style={{ display: "flex", fontSize: 22, color: "#9aa8ba" }}>Research only, not financial advice. HSF Score ranks setups, not a probability of profit.</div>
      </div>
    ),
    size,
  );
}
