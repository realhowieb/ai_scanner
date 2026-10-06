import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Sora } from "next/font/google";
import type { ReactNode } from "react";

import "./globals.css";

const sora = Sora({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-display", display: "swap" });
const plex = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans", display: "swap" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "HSFinest.AI", template: "%s · HSFinest.AI" },
  description: "HSF AI stock scanner: ranked setups, custom scans and stock intelligence.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0B0F14" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sora.variable} ${plex.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
