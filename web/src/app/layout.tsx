import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";

import { UsageAnalytics } from "@/components/UsageAnalytics";
import { ErrorReporter } from "@/components/ErrorReporter";
import { OfflineReady } from "@/components/OfflineReady";
import { STALE_BUILD_SCRIPT } from "@/lib/staleBuild";

import "./globals.css";

// Self-hosted (latin subset, from Fontsource 5.3.0, SIL OFL 1.1: see fonts/LICENSE.md) so the
// production build never fetches Google Fonts. A failed fetch broke `next build` under Turbopack.
const sora = localFont({
  src: "./fonts/sora-latin-wght-normal.woff2",
  weight: "500 700",
  variable: "--font-display",
  display: "swap",
});
const plex = localFont({
  src: [
    { path: "./fonts/ibm-plex-sans-latin-400-normal.woff2", weight: "400" },
    { path: "./fonts/ibm-plex-sans-latin-500-normal.woff2", weight: "500" },
    { path: "./fonts/ibm-plex-sans-latin-600-normal.woff2", weight: "600" },
  ],
  variable: "--font-sans",
  display: "swap",
});
const mono = localFont({
  src: [
    { path: "./fonts/ibm-plex-mono-latin-400-normal.woff2", weight: "400" },
    { path: "./fonts/ibm-plex-mono-latin-500-normal.woff2", weight: "500" },
    { path: "./fonts/ibm-plex-mono-latin-600-normal.woff2", weight: "600" },
  ],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "HSFinest.AI", template: "%s · HSFinest.AI" },
  description: "HSF AI stock scanner: ranked setups, custom scans and stock intelligence.",
  robots: { index: false, follow: false },
  // Link previews (iMessage, Telegram, X, Slack). Pages without their own image use app/opengraph-image.tsx.
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "https://app.hsfinest.com"),
  openGraph: {
    type: "website", siteName: "HSFinest.AI", title: "HSF AI Stock Scanner",
    description: "Ranked stock setups from scans of the whole U.S. market, with the reasons and risks behind each score.",
  },
  twitter: { card: "summary_large_image" },
  // Opened from the iPhone Home Screen: full screen, named "HSF", dark status bar.
  appleWebApp: { capable: true, title: "HSF", statusBarStyle: "black" },
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0B0F14" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sora.variable} ${plex.variable} ${mono.variable}`}>
      <head>
        {/* Reloads once when a deploy left this page pointing at files that are gone. */}
        <script dangerouslySetInnerHTML={{ __html: STALE_BUILD_SCRIPT }} />
      </head>
      <body>
        <UsageAnalytics />
        <ErrorReporter />
        <OfflineReady />
        {children}
      </body>
    </html>
  );
}
