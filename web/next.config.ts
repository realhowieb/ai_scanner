import type { NextConfig } from "next";
import { posthogHost } from "./src/lib/posthogConfig";

const analyticsOrigin = process.env.NEXT_PUBLIC_POSTHOG_KEY?.trim() ? ` ${posthogHost(process.env.NEXT_PUBLIC_POSTHOG_HOST)}` : "";
// Cloudflare Turnstile (sign-up and password reset) loads a script and an iframe from Cloudflare.
const turnstile = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY?.trim() ? " https://challenges.cloudflare.com" : "";

// Product data stays same-origin. Optional analytics may contact its configured ingestion origin.
const csp = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'" + turnstile + (process.env.NODE_ENV === "development" ? " 'unsafe-eval'" : ""),
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'" + analyticsOrigin + turnstile,
  ...(turnstile ? ["frame-src" + turnstile] : []),
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
].join("; ");

const nextConfig: NextConfig = {
  poweredByHeader: false,
  agentRules: false, // don't write AGENTS.md / CLAUDE.md into web/ on `next dev`
  // `npm run dev` only serves its dev scripts to localhost unless the host is listed;
  // without this, opening the dev site at 127.0.0.1 or a LAN address (a phone on the
  // same Wi-Fi) left the page unhydrated and the Sign in button disabled.
  // Add more with HSF_DEV_ORIGINS=192.168.1.20,my-laptop.local
  allowedDevOrigins: ["127.0.0.1", ...(process.env.HSF_DEV_ORIGINS || "").split(",").map((h) => h.trim()).filter(Boolean)],
  reactStrictMode: true,
  // Browser source maps only when they'll be uploaded to Sentry (scripts/sentry-sourcemaps.mjs
  // uploads, then deletes them from the build so they're never served).
  productionBrowserSourceMaps: Boolean(process.env.SENTRY_AUTH_TOKEN?.trim()),
  // Crash reports carry the build's commit so Sentry can match them to those maps.
  env: { NEXT_PUBLIC_HSF_RELEASE: process.env.RENDER_GIT_COMMIT?.trim() || process.env.HSF_RELEASE?.trim() || "" },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Content-Security-Policy", value: csp },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
          { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
        ],
      },
      {
        // The service worker must be re-checked on every visit so fixes reach browsers.
        source: "/sw.js",
        headers: [{ key: "Cache-Control", value: "no-cache" }],
      },
    ];
  },
};

export default nextConfig;
