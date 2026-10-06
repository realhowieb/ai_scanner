// Server-only settings. Nothing here reaches the browser bundle: no NEXT_PUBLIC_ prefix.
import "server-only";

export function apiBaseUrl(): string {
  const raw = (process.env.HSF_API_BASE_URL || "https://hsf-api.onrender.com").trim().replace(/\/+$/, "");
  const url = new URL(raw);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && isLoopback(url.hostname))) {
    throw new Error("HSF_API_BASE_URL must be https (http only for localhost)");
  }
  return raw;
}

function isLoopback(host: string): boolean {
  return host === "localhost" || host === "127.0.0.1" || host === "[::1]";
}

// The API can take ~45 s to answer its first request after sleeping (Render free plan).
export const UPSTREAM_TIMEOUT_MS = 75_000;
