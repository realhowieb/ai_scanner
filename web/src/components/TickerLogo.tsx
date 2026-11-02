"use client";

// Small company logo before a ticker. The image comes through our own /api/logo proxy
// (the CSP allows only same-origin images). Once it loads, the browser checks it the
// way the server does for the earnings email (api/logo_check.py): a white logo moves to
// a dark chip, and a photo or missing image shows the ticker's first letter instead.
import { useState } from "react";

type Kind = "pending" | "standard" | "light" | "none";
const seen = new Map<string, Kind>();
const SYMBOL = /^[A-Z]{1,6}([.-][A-Z]{1,2})?$/;

const LIGHT_LUMA = 0.82;
const TRANSPARENT_SHARE = 0.05;
const PHOTO_TOP_SHARE = 0.3;   // opaque, and the most common 4-bit color covers less than this = a photo

/** standard / light / none for a loaded logo image. Exported for tests. */
export function classifyPixels(data: Uint8ClampedArray): Exclude<Kind, "pending"> {
  let visible = 0, luma = 0;
  const total = data.length / 4;
  const counts = new Map<number, number>();
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]!, g = data[i + 1]!, b = data[i + 2]!, a = data[i + 3]!;
    const key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4);
    counts.set(key, (counts.get(key) ?? 0) + 1);
    if (a >= 128) { visible += 1; luma += 0.2126 * r + 0.7152 * g + 0.0722 * b; }
  }
  if (!visible) return "none";
  if (1 - visible / total < TRANSPARENT_SHARE) return Math.max(...counts.values()) / total < PHOTO_TOP_SHARE ? "none" : "standard";
  return luma / (255 * visible) > LIGHT_LUMA ? "light" : "standard";
}

function check(img: HTMLImageElement): Exclude<Kind, "pending"> {
  try {
    const size = 48;
    const canvas = document.createElement("canvas");
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return "standard";
    ctx.drawImage(img, 0, 0, size, size);
    return classifyPixels(ctx.getImageData(0, 0, size, size).data);
  } catch {
    return "standard";
  }
}

export function TickerLogo({ ticker, size = 20 }: { ticker: string; size?: number }) {
  const t = ticker.trim().toUpperCase();
  const [kind, setKind] = useState<Kind>(() => seen.get(t) ?? (SYMBOL.test(t) ? "pending" : "none"));
  const set = (k: Exclude<Kind, "pending">) => { seen.set(t, k); setKind(k); };
  const style = { width: size, height: size };
  if (kind === "none") return <span className="tk-logo tk-letter" style={{ ...style, fontSize: Math.max(9, Math.round(size / 2)) }} aria-hidden="true">{t.slice(0, 1)}</span>;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img className={`tk-logo${kind === "light" ? " tk-logo-dark" : ""}${kind === "pending" ? " tk-logo-pending" : ""}`}
      src={`/api/logo/${encodeURIComponent(t)}`} alt="" width={size} height={size} loading="lazy" decoding="async"
      style={style} onLoad={(e) => { if (kind === "pending") set(check(e.currentTarget)); }} onError={() => set("none")} />
  );
}
