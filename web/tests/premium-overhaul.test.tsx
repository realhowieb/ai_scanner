import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MarketStatus } from "@/components/MarketStatus";
import { SectionNav } from "@/components/SectionNav";
import { Card, ScoreBar, Stat, StatGrid } from "@/components/ui";
import { clearApiCache } from "@/hooks/useApi";

import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
beforeEach(() => { clearApiCache(); fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("header market status", () => {
  it("shows the session the API reports, as words and not color alone", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ quotes: [], phase: "afterhours" }));
    render(<MarketStatus />);
    const status = await screen.findByText("After hours");
    expect(status).toHaveClass("market-status", "tone-amber");
    expect(status.getAttribute("title")).toMatch(/not a live feed/);
  });

  it("stays hidden when the API doesn't say (no guessing from the clock)", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ quotes: [] }));
    const { container } = render(<MarketStatus />);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });
});

describe("source tags", () => {
  it("label what kind of information a card holds without changing its heading", () => {
    render(<>
      <Card title="Top setups" id="top" source="ranking"><p>rows</p></Card>
      <Card title="AI market narrative" id="ai" source="ai"><p>text</p></Card>
      <Card title="PreBreakout picks" id="pb" source="model"><p>picks</p></Card>
    </>);
    expect(screen.getByRole("heading", { name: "Top setups" })).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Top setups" })).getByText("HSF ranking")).toHaveAttribute("title", expect.stringMatching(/not a probability of profit/));
    expect(within(screen.getByRole("region", { name: "AI market narrative" })).getByText("AI-written")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "PreBreakout picks" })).getByText("Model estimate")).toHaveAttribute("title", expect.stringMatching(/not the HSF Score/));
  });
});

describe("shared pieces", () => {
  it("KPI tiles pair each figure with its label and note", () => {
    render(<StatGrid label="Signal counts"><Stat label="Still pending" value="286" note="not counted as wins or losses" /></StatGrid>);
    const grid = screen.getByLabelText("Signal counts");
    expect(within(grid).getByText("Still pending").tagName).toBe("DT");
    expect(within(grid).getByText("286").tagName).toBe("DD");
    expect(within(grid).getByText("not counted as wins or losses")).toBeInTheDocument();
  });

  it("score bars mark strong, ranked and weak scores", () => {
    const { container } = render(<><ScoreBar score={82} /><ScoreBar score={55} /><ScoreBar score={12} /><ScoreBar score={null} /></>);
    expect([...container.querySelectorAll(".scorebar")].map((b) => b.getAttribute("data-tier"))).toEqual(["strong", "ranked", "weak", "weak"]);
  });

  it("section links show only for sections on the page", async () => {
    render(<main id="main"><SectionNav label="AAPL sections" items={[["price", "Chart"], ["news", "News"], ["filings", "Filings"]]} />
      <section id="price" /><section id="news" /></main>);
    const nav = await screen.findByRole("navigation", { name: "AAPL sections" });
    expect(within(nav).getAllByRole("link").map((a) => [a.textContent, a.getAttribute("href")])).toEqual([["Chart", "#price"], ["News", "#news"]]);
  });
});
