import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PriceTape } from "@/features/PriceTape";

import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("Price strip", () => {
  it("scrolls each quote with its change, read once by screen readers", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ quotes: [{ symbol: "DIA", last: 511, chg_pct: -0.69 }, { symbol: "AAPL", last: 336.58, chg_pct: 0.86 }] }));
    render(<PriceTape />);
    const tape = await screen.findByRole("region", { name: "Market prices" });
    expect(tape).toHaveTextContent("DIA 511.00 -0.69%");
    expect(screen.getAllByText("+0.86%")[0]).toHaveClass("up");
    expect(tape.querySelectorAll('[aria-hidden="true"] .tape-item')).toHaveLength(2);
    expect(new URL((fetchMock.mock.calls[0]![0] as Request).url).pathname).toMatch(/\/v1\/market\/tape$/);
  });

  it("stays hidden without quotes or when the call fails", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ quotes: [] }));
    const { container, unmount } = render(<PriceTape />);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
    unmount();
    fetchMock.mockResolvedValue(jsonResponse({ detail: "down" }, 503));
    const r = render(<PriceTape />);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(r.container).toBeEmptyDOMElement();
  });
});
