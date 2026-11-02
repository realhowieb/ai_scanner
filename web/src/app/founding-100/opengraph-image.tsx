import { ImageResponse } from "next/og";

import { loadPublicFounding } from "@/server/publicFounder";

export const alt = "The Founding 100 of HSFinest.com: seats left";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const usd = (c: number) => `$${Math.round(c / 100).toLocaleString("en-US")}`;

/** The link preview for /founding-100: live seats left per tier. */
export default async function Image() {
  const d = await loadPublicFounding();
  const tiers = d?.tiers ?? [];
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between",
        background: "#1c1810", color: "#e7edf5", padding: "60px 72px", fontFamily: "sans-serif" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 34, fontWeight: 700 }}>
          <div style={{ display: "flex" }}><span>HSFinest</span><span style={{ color: "#4cc2ff" }}>.com</span></div>
          <div style={{ display: "flex", fontSize: 28, color: "#e3b55a" }}>Founding 100</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", fontSize: 72, fontWeight: 700, lineHeight: 1.1 }}>
            {d ? (d.remaining >= 100 ? "All 100 seats are open." : d.remaining > 0 ? `${d.remaining} of 100 seats left.` : "Sold out.") : "The Founding 100."}
          </div>
          <div style={{ display: "flex", fontSize: 34, color: "#b9c6d6", marginTop: 16 }}>One payment, five years. Numbered seats. No renewals.</div>
        </div>
        <div style={{ display: "flex", gap: 24 }}>
          {tiers.map((t) => (
            <div key={t.tier} style={{ display: "flex", alignItems: "center", gap: 16, background: "#2a2416", border: "2px solid #5a4a26",
              borderRadius: 16, padding: "16px 24px", fontSize: 30 }}>
              <span style={{ fontWeight: 700 }}>{t.label}</span>
              <span style={{ color: "#e3b55a", fontWeight: 700 }}>{usd(t.price_cents)}</span>
              <span style={{ color: "#b9c6d6", fontSize: 24 }}>{t.remaining} of {t.capacity} left</span>
            </div>
          ))}
        </div>
      </div>
    ),
    size,
  );
}
