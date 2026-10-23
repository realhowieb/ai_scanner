import { ImageResponse } from "next/og";

export const alt = "HSFinest.AI: ranked stock setups from scans of the whole U.S. market";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const SAMPLE = [["NVDA", "Breakout", 88], ["AMD", "Gapper", 76], ["SHOP", "Golden cross", 71]] as const;

/** The link preview for the landing page and every page without its own image. */
export default function Image() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between",
        background: "#0b0f14", color: "#e7edf5", padding: "64px 72px", fontFamily: "sans-serif" }}>
        <div style={{ display: "flex", fontSize: 40, fontWeight: 700 }}><span>HSFinest</span><span style={{ color: "#4cc2ff" }}>.AI</span></div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", fontSize: 64, fontWeight: 700, lineHeight: 1.1 }}>The whole U.S. market, ranked.</div>
          <div style={{ display: "flex", fontSize: 30, color: "#b9c6d6", marginTop: 18 }}>Scored setups, the reasons behind them, and alerts when they change.</div>
        </div>
        <div style={{ display: "flex", gap: 20 }}>
          {SAMPLE.map(([t, setup, score]) => (
            <div key={t} style={{ display: "flex", alignItems: "center", gap: 16, background: "#121821", border: "2px solid #243042",
              borderRadius: 16, padding: "16px 22px" }}>
              <span style={{ fontSize: 30, fontWeight: 700 }}>{t}</span>
              <span style={{ fontSize: 22, color: "#9aa8ba" }}>{setup}</span>
              <span style={{ fontSize: 28, fontWeight: 700, color: "#0b0f14", background: "#3dd68c", borderRadius: 10, padding: "4px 12px" }}>{score}</span>
            </div>
          ))}
        </div>
      </div>
    ),
    size,
  );
}
