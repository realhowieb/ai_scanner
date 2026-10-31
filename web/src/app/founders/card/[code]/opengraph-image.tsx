import { ImageResponse } from "next/og";

import { monthYear } from "@/lib/founders";
import { loadFounderCard } from "@/server/publicFounder";

export const alt = "HSF AI founder card";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

/** The preview image for a shared founder card: number, tier and dates. Never a name or email. */
export default async function Image({ params }: { params: Promise<{ code: string }> }) {
  const c = await loadFounderCard((await params).code);
  const elite = c?.tier === "founder_elite";
  const accent = elite ? "#e7edf5" : "#e3b55a";
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between",
        background: elite ? "#1f2a38" : "#1c1810", color: "#e7edf5", padding: "60px 72px", fontFamily: "sans-serif" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 34, fontWeight: 700 }}>
          <div style={{ display: "flex" }}><span>HSFinest</span><span style={{ color: "#4cc2ff" }}>.AI</span></div>
          <div style={{ display: "flex", fontSize: 28, color: accent }}>Founding 100</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", fontSize: 40, color: "#b9c6d6" }}>{c?.label ?? "Founding member"}</div>
          <div style={{ display: "flex", fontSize: 180, fontWeight: 700, color: accent, lineHeight: 1 }}>{c?.number ? `#${c.number}` : "#"}</div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 28, color: "#b9c6d6" }}>
          <span>{c?.since ? `Member since ${monthYear(c.since)}` : "One of 100 founding members"}</span>
          <span>{c?.until ? `Valid to ${monthYear(c.until)}` : ""}</span>
        </div>
      </div>
    ),
    size,
  );
}
