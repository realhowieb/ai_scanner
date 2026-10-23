import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AskHSFView } from "@/features/AskHSF";
import { JournalCoach } from "@/features/JournalCoach";
import { ScanBuilder } from "@/features/ScanBuilder";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

const calls: { method: string; path: string; body: unknown }[] = [];
let routes: Record<string, (body: unknown) => unknown> = {};
beforeEach(() => {
  calls.length = 0;
  routes = {};
  vi.stubGlobal("fetch", async (req: Request) => {
    const path = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
    const text = req.method === "GET" ? "" : await req.text();
    const body = text ? JSON.parse(text) : null;
    calls.push({ method: req.method, path, body });
    const h = routes[`${req.method} ${path}`];
    return h ? jsonResponse(h(body)) : jsonResponse({ detail: "Not Found" }, 404);
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Ask HSF (Premium)", () => {
  it("sends the conversation and shows the answer", async () => {
    routes["POST /v1/ai/ask"] = () => ({ answer: "AAA leads your list." });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("premium")}><AskHSFView /></SessionProvider>);
    await u.type(screen.getByRole("textbox", { name: "Your question" }), "How is $AAA?");
    await u.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByText("AAA leads your list.")).toBeInTheDocument();
    expect(calls.find((c) => c.path === "/v1/ai/ask")?.body).toEqual({ messages: [{ role: "user", content: "How is $AAA?" }] });
  });

  it("is locked below Premium", () => {
    render(<SessionProvider initialMe={me("pro")}><AskHSFView /></SessionProvider>);
    expect(screen.getByText("Ask HSF is part of Premium")).toBeInTheDocument();
  });
});

describe("Scan builder (Premium)", () => {
  it("fills the form from the AI settings", async () => {
    routes["POST /v1/ai/scan-builder"] = () => ({ view: { name: "AI scan", universe: "nasdaq", ticker: "", earnings_window: "today", session: "regular",
      top_n: 50, gap: true, unusual: false, score_all: false, min_price: 5 }, explanation: "NASDAQ gappers over $5." });
    const onApply = vi.fn();
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("premium")}><ScanBuilder onApply={onApply} /></SessionProvider>);
    await u.type(screen.getByRole("textbox", { name: /Describe your scan/ }), "nasdaq gappers over $5");
    await u.click(screen.getByRole("button", { name: "Fill in settings" }));
    expect(await screen.findByText(/NASDAQ gappers over \$5\. Nothing runs/)).toBeInTheDocument();
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({ universe: "nasdaq", top_n: 50, min_price: 5 }));
  });

  it("is hidden below Premium", () => {
    const { container } = render(<SessionProvider initialMe={me("pro")}><ScanBuilder onApply={vi.fn()} /></SessionProvider>);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Journal coach (Premium)", () => {
  it("asks for more trades when the journal is short", async () => {
    routes["POST /v1/ai/journal-coach"] = () => ({ closed: 2, min_trades: 5, text: null });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("premium")}><JournalCoach /></SessionProvider>);
    await u.click(screen.getByRole("button", { name: "Coach me" }));
    expect(await screen.findByText("Close at least 5 trades for coaching (you have 2).")).toBeInTheDocument();
  });

  it("is locked below Premium", () => {
    render(<SessionProvider initialMe={me("pro")}><JournalCoach /></SessionProvider>);
    expect(screen.getByText("The AI journal coach is part of Premium")).toBeInTheDocument();
  });
});
