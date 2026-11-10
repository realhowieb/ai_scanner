import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ScannerView } from "@/features/ScannerView";
import { RankMove, sinceScan } from "@/features/SetupTable";
import { SessionProvider } from "@/session/SessionProvider";

import { me, setup, stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace, push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/scanner",
}));

const fetchMock = vi.fn();
const urls = () => fetchMock.mock.calls.map((c) => new URL((c[0] as Request).url));
beforeEach(() => {
  search = new URLSearchParams();
  replace.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", (request: Request) => fetchMock(request));
});
afterEach(() => vi.unstubAllGlobals());

const latest = {
  scan_at: new Date().toISOString(), previous_scan_at: new Date(Date.now() - 3600_000).toISOString(),
  total: 3, max_results: 100, limited: false,
  sectors: [{ sector: "Technology", count: 2 }, { sector: "Finance", count: 1 }],
  setups: [
    setup("AAA", 88, { rank: 1, previous_rank: 9, sector: "Technology" }),
    setup("BBB", 70, { rank: 2, previous_rank: null, sector: "Finance" }),
    setup("CCC", 60, { rank: 3, previous_rank: 2, sector: "Technology" }),
  ],
};
const route = () => fetchMock.mockImplementation(async (req: Request) => {
  const p = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
  if (p === "/v1/scans/latest") return jsonResponse(latest);
  if (p === "/v1/stocks/AAA") return jsonResponse(stockDetail({ ticker: "AAA", reasons: ["Broke out on 3x volume"], risks: ["Extended above the 20-day average"] }));
  if (p === "/v1/me/research-preferences") return jsonResponse({ scanner_views: [], custom_views: [], table: { columns: ["setup", "spark", "last", "change", "gap", "rvol", "prob"], compact: false }, quiet_hours: { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" } });
  return jsonResponse({ detail: `no fake for ${p}` }, 404);
});
const wrap = () => render(<SessionProvider initialMe={me("pro")}><ScannerView /></SessionProvider>);

describe("Rank moves", () => {
  it("marks new names and moves of 3+ places, and nothing without a previous scan", () => {
    const { rerender } = render(<RankMove row={{ rank: 2, previous_rank: null }} compared />);
    expect(screen.getByText("New")).toBeInTheDocument();
    rerender(<RankMove row={{ rank: 1, previous_rank: 9 }} compared />);
    expect(screen.getByLabelText("Up 8 places since the previous scan (was #9)")).toHaveTextContent("▲8");
    rerender(<RankMove row={{ rank: 6, previous_rank: 2 }} compared />);
    expect(screen.getByText("▼4")).toBeInTheDocument();
    rerender(<RankMove row={{ rank: 3, previous_rank: 2 }} compared />);
    expect(screen.queryByText(/[▲▼]/)).not.toBeInTheDocument();
    rerender(<RankMove row={{ rank: 2, previous_rank: null }} compared={false} />);
    expect(screen.queryByText("New")).not.toBeInTheDocument();
  });

  it("measures the live price against the scan's", () => {
    expect(sinceScan({ last: 105, scan_last: 100 })).toBeCloseTo(5);
    expect(sinceScan({ last: 105, scan_last: null })).toBeNull();
  });
});

describe("Scanner moves, sectors and quick view", () => {
  it("shows New / ▲▼ badges, the sector column and the sector filter", async () => {
    route();
    wrap();
    const table = await screen.findByRole("region", { name: /Ranked setups table/ });
    expect(within(table).getByText("New")).toBeInTheDocument();
    expect(within(table).getByText("▲8")).toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: "Sector" })).toBeInTheDocument();
    expect(within(table).getAllByText("Technology")).toHaveLength(2);
    expect(screen.getByText(/New and ▲▼ compare each rank with the/)).toBeInTheDocument();
    await userEvent.setup().selectOptions(screen.getByRole("combobox", { name: "Sector" }), "Finance");
    expect(replace).toHaveBeenCalledWith("/scanner?sector=Finance", { scroll: false });
  });

  it("sends the sector filter to the API", async () => {
    search = new URLSearchParams("sector=Finance");
    route();
    wrap();
    await waitFor(() => expect(urls().some((u) => u.pathname.endsWith("/v1/scans/latest"))).toBe(true));
    expect(urls().find((u) => u.pathname.endsWith("/v1/scans/latest"))!.searchParams.get("sector")).toBe("Finance");
  });

  it("opens a quick view with why it ranks and a link to the full page", async () => {
    route();
    wrap();
    const table = await screen.findByRole("region", { name: /Ranked setups table/ });
    await userEvent.setup().click(within(table).getByRole("button", { name: "Quick view AAA" }));
    const panel = await screen.findByRole("dialog", { name: "AAA" });
    expect(await within(panel).findByText("Broke out on 3x volume")).toBeInTheDocument();
    expect(within(panel).getByText("Extended above the 20-day average")).toBeInTheDocument();
    expect(within(panel).getByRole("link", { name: "Open full page" })).toHaveAttribute("href", "/stocks/AAA");
  });
});

describe("Sector in Table display", () => {
  it("lists Sector so it can be turned off", async () => {
    route();
    wrap();
    await screen.findByRole("region", { name: /Ranked setups table/ });
    await userEvent.setup().click(screen.getByText("Table display"));
    expect(screen.getByRole("checkbox", { name: "Sector" })).toBeChecked();
    expect(screen.getByRole("button", { name: "Move Sector later" })).toBeInTheDocument();
  });
});
