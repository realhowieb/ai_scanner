// The image for a weekly social post (services/social_posts.py sends it to Buffer as
// <site>/social/<kind>/<week>.png). Public: Buffer and the networks fetch it signed out.
import { ImageResponse } from "next/og";

import { upstream } from "@/server/upstream";

export const dynamic = "force-dynamic";

const KINDS = new Set(["earnings", "spotlight", "product", "explainer", "founding", "recap"]);
const FILE_RE = /^(\d{4}-W\d{2})\.png$/;
const W = 1080;
const H = 1350;
const C = { bg: "#0b0f14", panel: "#121923", line: "#243042", text: "#e7edf5", dim: "#9aa8ba", accent: "#4cc2ff" };

type Name = { ticker: string; time?: string | null; name?: string | null; days?: number; setup?: string | null };
type Img = {
  kicker?: string; headline?: string; sub?: string; points?: string[];
  days?: { day: string; names: Name[] }[]; seats?: { label: string; price: string; left: number; of: number }[]; names?: Name[];
};

function Chip({ n }: { n: Name }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 10, background: C.bg, border: `2px solid ${C.line}`, borderRadius: 14, padding: "8px 18px" }}>
      <span style={{ fontSize: 36, fontWeight: 700 }}>${n.ticker}</span>
      {n.time ? <span style={{ fontSize: 20, color: C.dim }}>{n.time.toUpperCase()}</span> : null}
    </div>
  );
}

function Body({ img }: { img: Img }) {
  const box = { display: "flex", alignItems: "center", gap: 28, background: C.panel, border: `2px solid ${C.line}`, borderRadius: 22, padding: "22px 30px" } as const;
  if (img.days) return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {img.days.map((d) => (
        <div key={d.day} style={box}>
          <span style={{ width: 90, fontSize: 32, fontWeight: 700, color: C.dim }}>{d.day}</span>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>{d.names.map((n) => <Chip key={n.ticker} n={n} />)}</div>
        </div>
      ))}
    </div>
  );
  if (img.seats) return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {img.seats.map((s) => (
        <div key={s.label} style={{ ...box, justifyContent: "space-between", padding: "36px 40px" }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontSize: 44, fontWeight: 700 }}>{s.label}</span>
            <span style={{ fontSize: 30, color: C.dim, marginTop: 8 }}>{s.price}, one payment</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
            <span style={{ fontSize: 64, fontWeight: 700, color: C.accent }}>{s.left}</span>
            <span style={{ fontSize: 24, color: C.dim }}>of {s.of} left</span>
          </div>
        </div>
      ))}
    </div>
  );
  if (img.names) return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {img.names.map((n, i) => (
        <div key={n.ticker} style={{ ...box, padding: "30px 36px" }}>
          <span style={{ fontSize: 40, fontWeight: 700, color: C.accent, width: 60 }}>{i + 1}</span>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontSize: 48, fontWeight: 700 }}>${n.ticker}</span>
            <span style={{ fontSize: 26, color: C.dim, marginTop: 6 }}>
              {[n.name, n.days ? `${n.days} day${n.days === 1 ? "" : "s"} in the top 10` : null, n.setup].filter(Boolean).join(" · ").slice(0, 70)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {(img.points ?? []).map((p) => (
        <div key={p} style={{ ...box, padding: "30px 36px" }}>
          <span style={{ width: 18, height: 18, borderRadius: 9, background: C.accent }} />
          <span style={{ fontSize: 40, fontWeight: 600 }}>{p}</span>
        </div>
      ))}
    </div>
  );
}

export async function GET(_req: Request, ctx: { params: Promise<{ kind: string; file: string }> }) {
  const { kind, file } = await ctx.params;
  const m = FILE_RE.exec(file || "");
  if (!KINDS.has(kind) || !m) return new Response("Not found", { status: 404 });
  let img: Img;
  try {
    const res = await upstream(`/v1/public/social/${kind}/${m[1]}`, { method: "GET", headers: { accept: "application/json" } });
    if (!res.ok) return new Response("Not found", { status: 404 });
    img = ((await res.json()) as { image?: Img }).image ?? {};
  } catch {
    return new Response("Unavailable", { status: 503 });
  }
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: C.bg, color: C.text,
        padding: "80px 80px 70px", fontFamily: "sans-serif",
        backgroundImage: "radial-gradient(circle at 90% 0%, rgba(76,194,255,0.18), transparent 45%)" }}>
        <div style={{ display: "flex", fontSize: 40, fontWeight: 700 }}><span>HSFinest</span><span style={{ color: C.accent }}>.com</span></div>
        <div style={{ display: "flex", marginTop: 64, fontSize: 28, letterSpacing: 4, color: C.accent, fontWeight: 700 }}>{(img.kicker ?? "").toUpperCase()}</div>
        <div style={{ display: "flex", marginTop: 14, fontSize: 84, fontWeight: 700, lineHeight: 1.05 }}>{img.headline ?? ""}</div>
        {img.sub ? <div style={{ display: "flex", marginTop: 18, fontSize: 30, color: C.dim }}>{img.sub}</div> : null}
        <div style={{ display: "flex", flexDirection: "column", marginTop: 52 }}><Body img={img} /></div>
        <div style={{ display: "flex", marginTop: "auto", justifyContent: "space-between", alignItems: "flex-end" }}>
          <span style={{ fontSize: 32, fontWeight: 700 }}>www.hsfinest.com</span>
          <span style={{ fontSize: 20, color: "#6f8399" }}>Research only, not financial advice.</span>
        </div>
      </div>
    ),
    { width: W, height: H, headers: { "cache-control": "public, max-age=3600" } },
  );
}
