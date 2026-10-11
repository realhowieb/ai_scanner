import type { MetadataRoute } from "next";

// Lets phones add HSF to the Home Screen; iPhones only allow notifications from there.
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "HSFinest.AI",
    short_name: "HSF",
    start_url: "/today",
    display: "standalone",
    background_color: "#0b0f14",
    theme_color: "#0b0f14",
    icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml" }],
  };
}
