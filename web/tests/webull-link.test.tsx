import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { OpenInWebull, webullHref } from "@/features/OpenInWebull";
import type { Upstream } from "@/server/bff";
import { WEBULL_FALLBACK, webullTarget } from "@/server/webullLink";

const answer = (status: number, body: unknown): Upstream => vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));

describe("webullTarget", () => {
  it("redirects to the stock's Webull page from the API", async () => {
    const up = answer(200, { url: "https://www.webull.com/quote/nyse-brk-b", exact: true });
    expect(await webullTarget("brk-b", up)).toBe("https://www.webull.com/quote/nyse-brk-b");
    expect(vi.mocked(up).mock.calls[0]?.[0]).toBe("/v1/public/webull-link/BRK-B");
  });
  it("falls back to Webull's markets page on errors and bad tickers", async () => {
    expect(await webullTarget("AAPL", answer(503, {}))).toBe(WEBULL_FALLBACK);
    expect(await webullTarget("AAPL", vi.fn(async () => { throw new Error("down"); }))).toBe(WEBULL_FALLBACK);
    const up = answer(200, {});
    expect(await webullTarget("../etc", up)).toBe(WEBULL_FALLBACK);
    expect(up).not.toHaveBeenCalled();
  });
  it("never redirects off webull.com", async () => {
    expect(await webullTarget("AAPL", answer(200, { url: "https://evil.example/quote" }))).toBe(WEBULL_FALLBACK);
    expect(await webullTarget("AAPL", answer(200, { url: "javascript:alert(1)" }))).toBe(WEBULL_FALLBACK);
  });
});

describe("OpenInWebull", () => {
  it("is a new-tab link to /go/webull", () => {
    render(<><OpenInWebull ticker="aapl" /><OpenInWebull ticker="spy" compact /></>);
    const full = screen.getByRole("link", { name: "Open in Webull" });
    expect(full.getAttribute("href")).toBe("/go/webull/AAPL");
    expect(full.getAttribute("target")).toBe("_blank");
    expect(full.getAttribute("rel")).toContain("noopener");
    expect(screen.getByRole("link", { name: "Open SPY in Webull" }).getAttribute("href")).toBe(webullHref("SPY"));
  });
});
