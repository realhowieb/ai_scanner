import type { MetadataRoute } from "next";

// Lets phones add HSF to the Home Screen; iPhones only allow notifications from there.
// iPhones ignore SVG icons, so the PNGs (and app/apple-icon.png) are what the Home Screen shows.
export default function manifest(): MetadataRoute.Manifest {
  return {
    id: "/today",
    name: "HSFinest.AI",
    short_name: "HSF",
    start_url: "/today",
    scope: "/",
    display: "standalone",
    background_color: "#0b0f14",
    theme_color: "#0b0f14",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml" },
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
