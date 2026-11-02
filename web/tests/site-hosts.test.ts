import { describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

import { appHandoffUrl } from "@/components/MarketingHandoff";
import { captureAttribution, handoffParams } from "@/lib/funnel";
import { isMarketingHost, marketingRoute } from "@/lib/siteHosts";
import { config, proxy } from "@/proxy";

function hit(url: string, host?: string) {
  const req = new NextRequest(url, { headers: host ? { host } : {} });
  return proxy(req);
}

describe("marketing host (www.hsfinest.com)", () => {
  it("knows its hosts", () => {
    expect(isMarketingHost("www.hsfinest.com")).toBe(true);
    expect(isMarketingHost("HSFinest.com")).toBe(true);
    expect(isMarketingHost("app.hsfinest.com")).toBe(false);
    expect(isMarketingHost("hsf-web-v1.onrender.com")).toBe(false);
  });

  it("serves public pages and their files", () => {
    for (const p of ["/", "/pricing", "/how-hsf-works", "/demo", "/weekly", "/weekly/2026-W44", "/founders",
      "/founders/card/abcd", "/founding-100/terms", "/terms", "/privacy", "/share/AAPL", "/api/public/v1/plans",
      "/api/public/v1/events", "/opengraph-image", "/robots.txt", "/sitemap.xml", "/landing/scanner-current.jpg",
      "/icon-192.png", "/subscribe/confirm", "/unsubscribe"]) {
      expect(marketingRoute(p), p).toBe("serve");
    }
  });

  it("sends old music pages to music.hsfinest.com and the rest to the app", () => {
    for (const p of ["/shop", "/shop/sunglasses", "/blog-hsf", "/music", "/events", "/services"]) expect(marketingRoute(p), p).toBe("music");
    for (const p of ["/login", "/signup", "/today", "/scanner", "/account", "/api/auth/login", "/api/hsf/v1/me", "/pricingx", "/musician"]) {
      expect(marketingRoute(p), p).toBe("app");
    }
  });

  it("redirects through the proxy, keeping the query", () => {
    const toApp = hit("https://www.hsfinest.com/signup?utm_source=instagram", "www.hsfinest.com");
    expect(toApp.status).toBe(307);
    expect(toApp.headers.get("location")).toBe("https://app.hsfinest.com/signup?utm_source=instagram");
    const toMusic = hit("https://hsfinest.com/shop", "hsfinest.com");
    expect(toMusic.status).toBe(301);
    expect(toMusic.headers.get("location")).toBe("https://music.hsfinest.com/shop");
    expect(hit("https://www.hsfinest.com/pricing", "www.hsfinest.com").headers.get("location")).toBeNull();
  });

  it("leaves the app host as it was", () => {
    // Public pages on the app host are never sent to sign-in, even if the proxy runs for them.
    expect(hit("https://app.hsfinest.com/pricing", "app.hsfinest.com").headers.get("location")).toBeNull();
    expect(hit("https://app.hsfinest.com/today", "app.hsfinest.com").headers.get("location")).toContain("/login?next=%2Ftoday");
    expect(hit("https://app.hsfinest.com/", "app.hsfinest.com").headers.get("location")).toBeNull();
  });

  it("matches every marketing-host path, by host", () => {
    const entry = config.matcher[0] as { source: string; has: { type: string; value: string }[] };
    expect(entry.has[0]).toEqual({ type: "host", value: "(www\\.)?hsfinest\\.com" });
  });
});

describe("hand-off to the app", () => {
  it("rewrites app links to app.hsfinest.com with the visitor's attribution", () => {
    const params = handoffParams({ utm_source: "instagram", ref: "abcd1234", referrer: "https://google.com/" });
    expect(appHandoffUrl("/signup", "https://www.hsfinest.com/pricing", params))
      .toBe("https://app.hsfinest.com/signup?utm_source=instagram&ref=abcd1234&hsf_referrer=https%3A%2F%2Fgoogle.com%2F");
    expect(appHandoffUrl("/login?next=%2Ftoday", "https://www.hsfinest.com/", new URLSearchParams()))
      .toBe("https://app.hsfinest.com/login?next=%2Ftoday");
  });

  it("leaves public and outside links alone", () => {
    const p = new URLSearchParams();
    expect(appHandoffUrl("/pricing", "https://www.hsfinest.com/", p)).toBeNull();
    expect(appHandoffUrl("https://music.hsfinest.com", "https://www.hsfinest.com/", p)).toBeNull();
  });

  it("the app reads the handed-over referrer", () => {
    localStorage.clear();
    const a = captureAttribution("https://app.hsfinest.com/signup?utm_source=threads&hsf_referrer=https%3A%2F%2Fwww.threads.net%2F", "");
    expect(a).toEqual({ utm_source: "threads", referrer: "https://www.threads.net/" });
    localStorage.clear();
  });
});

describe("music site link", () => {
  it("is in the public header menu", async () => {
    const { render, screen } = await import("@testing-library/react");
    const { createElement } = await import("react");
    const { PublicHeader } = await import("@/features/Landing");
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 401 })));
    render(createElement(PublicHeader));
    const link = screen.getByRole("link", { name: "Music & Shop" });
    expect(link.getAttribute("href")).toBe("https://music.hsfinest.com");
    vi.unstubAllGlobals();
  });
});
