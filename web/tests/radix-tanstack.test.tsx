import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { AppShell } from "@/components/AppShell";
import { Dialog } from "@/components/Dialog";
import { Hint } from "@/components/Hint";
import { SetupTable } from "@/features/SetupTable";
import { columnOrder } from "@/features/TablePreferences";
import { SessionProvider } from "@/session/SessionProvider";

import { me, setup } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/scanner",
}));

let preferences: Schemas["WorkspacePreferences"];
let patches: Schemas["WorkspacePatch"][];
const fetchMock = vi.fn(async (request: Request) => {
  const path = new URL(request.url).pathname.replace("/api/hsf", "");
  if (path === "/v1/me/research-preferences") {
    if (request.method === "PATCH") { const body = await request.json(); patches.push(body); preferences = { ...preferences, ...body }; }
    return jsonResponse(preferences);
  }
  if (path === "/v1/sparklines") return jsonResponse({ series: {} });
  return jsonResponse([]);
});
beforeEach(() => {
  patches = [];
  preferences = { scanner_views: [], custom_views: [], table: { columns: ["setup", "spark", "last", "change", "gap", "rvol", "prob"], compact: false },
    quiet_hours: { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" } };
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const headers = () => within(screen.getByRole("region", { name: /Ranked setups table/ })).getAllByRole("columnheader").map((h) => h.textContent?.replace("↓", "").trim());

describe("Radix dialog", () => {
  function Opener() {
    const [open, setOpen] = useState(false);
    return <><button type="button" onClick={() => setOpen(true)}>Open it</button><p>Page text</p>
      <Dialog open={open} title="Test dialog" onClose={() => setOpen(false)}><button type="button" data-autofocus>Inside</button></Dialog></>;
  }
  it("focuses the marked control, hides the page from screen readers, locks scroll, and returns focus on Escape", async () => {
    const u = userEvent.setup();
    render(<Opener />);
    const opener = screen.getByRole("button", { name: "Open it" });
    await u.click(opener);
    const dlg = screen.getByRole("dialog", { name: "Test dialog" });
    expect(within(dlg).getByRole("button", { name: "Inside" })).toHaveFocus();
    expect(screen.queryByRole("button", { name: "Open it" })).not.toBeInTheDocument(); // aria-hidden behind the modal
    expect(document.body).toHaveAttribute("data-scroll-locked");
    await u.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(opener).toHaveFocus());
  });
});

describe("Radix menus", () => {
  it("moves through the account menu with the arrow keys", async () => {
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AppShell><p>Body</p></AppShell></SessionProvider>);
    const button = screen.getByRole("button", { name: /Account/ });
    button.focus();
    await u.keyboard("{Enter}");
    const menu = await screen.findByRole("menu");
    const items = within(menu).getAllByRole("menuitem");
    await waitFor(() => expect(items[0]).toHaveFocus());
    await u.keyboard("{ArrowDown}");
    expect(items[1]).toHaveFocus();
    await u.keyboard("{End}");
    expect(within(menu).getByRole("menuitem", { name: "Sign out" })).toHaveFocus();
    await u.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(button).toHaveFocus();
  });
});

describe("Hint", () => {
  it("opens on tap or keyboard, not only on hover", async () => {
    const u = userEvent.setup();
    render(<Hint label="Relative volume">Today vs average.</Hint>);
    await u.click(screen.getByRole("button", { name: "About Relative volume" }));
    expect(await screen.findByRole("dialog")).toHaveTextContent("Today vs average.");
    await u.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});

describe("Scanner table (TanStack)", () => {
  const rows = [setup("AAA", 90), setup("BBB", 80)];
  it("sorts on the server from the headers and keeps the HSF rank column", async () => {
    const onSort = vi.fn();
    const { rerender } = render(<SessionProvider initialMe={me("premium")}><SetupTable rows={rows} premium sparks displayControls={false} sort="score" onSort={onSort} /></SessionProvider>);
    await waitFor(() => expect(headers()).toContain("Setup"));
    const score = screen.getByRole("columnheader", { name: /HSF Score/ });
    expect(score).toHaveAttribute("aria-sort", "descending");
    fireEvent.click(within(screen.getByRole("columnheader", { name: /Chg %/ })).getByRole("button", { name: /Chg %/ }));
    expect(onSort).toHaveBeenLastCalledWith("chg_pct");
    fireEvent.click(within(screen.getByRole("columnheader", { name: /PreBreakout/ })).getByRole("button", { name: /^PreBreakout/ }));
    expect(onSort).toHaveBeenLastCalledWith("prob");
    // Rows are shown in the order the API sent them; the server did the sorting.
    rerender(<SessionProvider initialMe={me("premium")}><SetupTable rows={[rows[1]!, rows[0]!]} premium sparks displayControls={false} numbered={false} sort="chg_pct" onSort={onSort} /></SessionProvider>);
    expect(screen.getByRole("columnheader", { name: /Chg %/ })).toHaveAttribute("aria-sort", "descending");
    expect(screen.getByRole("columnheader", { name: /HSF Score/ })).toHaveAttribute("aria-sort", "none");
    const firstRow = within(screen.getByRole("region", { name: /Ranked setups table/ })).getAllByRole("row")[1]!;
    expect(firstRow).toHaveTextContent("BBB");
    expect(within(screen.getByRole("columnheader", { name: /Last/ })).queryByRole("button")).not.toBeInTheDocument();
  });

  it("hides the PreBreakout sort below Premium and renders no sort buttons without onSort", async () => {
    render(<SessionProvider initialMe={me("pro")}><SetupTable rows={rows} premium={false} /></SessionProvider>);
    expect(screen.queryByRole("columnheader", { name: /PreBreakout/ })).not.toBeInTheDocument();
    expect(within(screen.getByRole("columnheader", { name: /HSF Score/ })).queryByRole("button", { name: /^HSF Score/ })).not.toBeInTheDocument();
  });

  it("shows columns in the saved order and saves a new order from Table display", async () => {
    preferences.table = { columns: ["setup", "rvol", "last", "change"], compact: false };
    render(<SessionProvider initialMe={me("pro")}><SetupTable rows={rows} premium={false} sparks /></SessionProvider>);
    await waitFor(() => expect(headers().slice(0, 7)).toEqual(["#", "Ticker", "Setup", "HSF Score", "RVOL", "Last", "Chg %"]));
    fireEvent.click(screen.getByText("Table display"));
    fireEvent.click(screen.getByRole("button", { name: "Move Last price earlier" }));
    fireEvent.click(screen.getByText("Save display"));
    await waitFor(() => expect(patches[0]?.table?.columns).toEqual(["setup", "last", "rvol", "change"]));
    await waitFor(() => expect(headers().slice(4, 7)).toEqual(["Last", "RVOL", "Chg %"]));
  });

  it("keeps an open Save dialog when the table re-renders", async () => {
    const u = userEvent.setup();
    const { rerender } = render(<SessionProvider initialMe={me("pro")}><SetupTable rows={rows} premium={false} sparks /></SessionProvider>);
    await u.click(screen.getAllByRole("button", { name: "Save AAA to a watchlist" })[0]!);
    rerender(<SessionProvider initialMe={me("pro")}><SetupTable rows={rows} premium={false} sparks offset={0} /></SessionProvider>);
    expect(screen.getByRole("dialog", { name: "Save AAA to a watchlist" })).toBeInTheDocument();
  });

  it("orders movable columns from the saved list", () => {
    expect(columnOrder(["setup", "gap", "spark"])).toEqual(["gap", "spark", "last", "change", "rvol", "prob"]);
  });
});
