import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SignupForm } from "@/features/AuthForms";
import { LiveSetupsPreview } from "@/features/LiveSetupsPreview";

import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/",
}));

let answer: Response | null = null;
beforeEach(() => {
  vi.stubGlobal("fetch", async (req: Request | string) => {
    const url = typeof req === "string" ? req : req.url;
    if (url.includes("/v1/public/top-setups") && answer) return answer.clone();
    return jsonResponse({ detail: "Not Found" }, 404);
  });
});
afterEach(() => { answer = null; vi.unstubAllGlobals(); });

describe("landing live preview", () => {
  it("lists the latest scan's top setups with share links and how many more are free", async () => {
    answer = jsonResponse({ scan_at: "2026-10-29T13:35:00Z", stale: false, ranked_total: 40, free_rows: 25,
      setups: [{ ticker: "AAA", hsf_score: 91, setup: "breakout", change_pct: 2.1 },
        { ticker: "BBB", hsf_score: 84, setup: "gapper", change_pct: -0.4 }] });
    render(<LiveSetupsPreview />);
    expect(await screen.findByRole("link", { name: "AAA" })).toHaveAttribute("href", "/share/AAA");
    expect(screen.getByText("Top setups from the latest scan")).toBeInTheDocument();   // an Oct 29 scan is not "today"
    expect(screen.getByText(/\+23 more ranked setups/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See all ranked setups free" })).toHaveAttribute("href", "/signup");
  });

  it("falls back to the sample card when the API can't answer", async () => {
    render(<LiveSetupsPreview />);
    await waitFor(() => expect(screen.getByText("Sample data · not live")).toBeInTheDocument());
  });
});

describe("sign-up agreement", () => {
  it("links the Terms of Service and Privacy Policy once they are published", () => {
    const { unmount } = render(<SignupForm terms />);
    expect(screen.getByRole("link", { name: "Terms of Service" })).toHaveAttribute("href", "/terms");
    expect(screen.getByRole("link", { name: "Privacy Policy" })).toHaveAttribute("href", "/privacy");
    unmount();
    render(<SignupForm />);
    expect(screen.queryByRole("link", { name: "Terms of Service" })).not.toBeInTheDocument();
    expect(screen.getByText(/educational\/informational purposes only/)).toBeInTheDocument();
  });
});
