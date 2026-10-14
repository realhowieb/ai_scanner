import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { AppShell } from "@/components/AppShell";
import { AdminView } from "@/features/AdminView";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/admin",
}));

const admin = (): Schemas["Me"] => ({ ...me("basic"), email: "boss@example.com", is_admin: true });
let users: Schemas["AdminUser"][] = [];
const calls: { method: string; path: string; body: unknown }[] = [];

async function handle(input: Request | string, init?: RequestInit): Promise<Response> {
  const req = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
  const path = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
  const text = req.method === "GET" ? "" : await req.text();
  const body = text ? JSON.parse(text) : null;
  calls.push({ method: req.method, path, body });
  if (req.method === "GET" && path === "/v1/admin/users") return jsonResponse({ users });
  if (req.method === "GET" && path === "/v1/admin/audit") return jsonResponse({ events: [{ at: "2026-10-14T12:00:00Z", actor: "boss@example.com", action: "grant admin", target: "x@example.com", details: "" }] });
  if (path === "/v1/admin/users/update") {
    const b = body as { username: string; tier?: string };
    users = users.map((u) => (u.username === b.username ? { ...u, tier: b.tier ?? u.tier } : u));
    return jsonResponse({ ok: true, message: `Updated ${b.username}.` });
  }
  if (path === "/v1/admin/users/role") return jsonResponse({ detail: "You can't remove your own admin role." }, 422);
  if (path === "/v1/admin/health") return jsonResponse({ operations: { system_status: "DEGRADED", generated_at: "2026-10-14T12:00:00Z", stale: false, workflows: [] },
    health_score: 80, human_action: "NONE", next_expected_scan: null, subsystems: [{ name: "Scanner", status: "HEALTHY", label: "ok", reason: "All slots ran" }],
    incidents: [], email: { status: "HEALTHY", day: "2026-10-14", jobs: [{ job: "digest", label: "Morning digest", status: "HEALTHY", detail: "sent 3" }] } });
  if (path === "/v1/admin/providers") return jsonResponse({ checked_at: "2026-10-14T12:00:00Z", earnings_cache: { last_refresh: null, dated_rows: 10 },
    checks: [{ name: "Database", status: "ok", detail: "neon", ms: 5 }, { name: "Billing service", status: "error", detail: "down", ms: null }] });
  if (req.method === "GET" && path === "/v1/admin/earnings-refresh") return jsonResponse({ state: "idle" });
  if (req.method === "POST" && path === "/v1/admin/earnings-refresh") return jsonResponse({ state: "done", symbols: 600, dated: 420, finished_at: "2026-10-14T12:00:00Z" });
  return jsonResponse({ detail: `no fake for ${req.method} ${path}` }, 404);
}

beforeEach(() => {
  calls.length = 0;
  users = [
    { username: "boss@example.com", full_name: "Boss", tier: "basic", is_active: true, is_admin: true, email_verified: true, created_at: null },
    { username: "pro@example.com", full_name: "Pro User", tier: "pro", is_active: true, is_admin: false, email_verified: false, created_at: null },
  ];
  vi.stubGlobal("fetch", vi.fn(handle));
});
afterEach(() => vi.unstubAllGlobals());

const wrap = (m: Schemas["Me"]) => render(<SessionProvider initialMe={m}><AdminView /></SessionProvider>);

describe("Admin page", () => {
  it("is for admins only, and only admins see it in the account menu", async () => {
    const u = userEvent.setup();
    const { unmount } = wrap(me("pro"));
    expect(screen.getByText("Admins only.")).toBeInTheDocument();
    expect(calls).toHaveLength(0);
    unmount();
    const shell = (m: Schemas["Me"]) => render(<SessionProvider initialMe={m}><AppShell><p>body</p></AppShell></SessionProvider>);
    const a = shell(me("pro"));
    await u.click(screen.getByRole("button", { name: /Account/ }));
    expect(within(screen.getByRole("menu")).queryByRole("menuitem", { name: "Admin" })).not.toBeInTheDocument();
    a.unmount();
    shell(admin());
    await u.click(screen.getByRole("button", { name: /Account/ }));
    expect(within(screen.getByRole("menu")).getByRole("menuitem", { name: "Admin" })).toHaveAttribute("href", "/admin");
  });

  it("lists accounts, changes a plan and shows refusals", async () => {
    const u = userEvent.setup();
    wrap(admin());
    expect(await screen.findByText("pro@example.com")).toBeInTheDocument();
    expect(screen.getByText("2 accounts · 1 admins · 1 unverified")).toBeInTheDocument();
    await u.type(screen.getByRole("searchbox"), "pro");
    expect(screen.queryByText("boss@example.com", { selector: "strong" })).not.toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Manage pro@example.com" }));
    const dialog = screen.getByRole("dialog");
    await u.selectOptions(within(dialog).getByRole("combobox"), "premium");
    await u.click(within(dialog).getByRole("button", { name: "Save changes" }));
    expect(await within(dialog).findByText("Updated pro@example.com.")).toBeInTheDocument();
    expect(calls.find((c) => c.path === "/v1/admin/users/update")?.body).toEqual({ username: "pro@example.com", tier: "premium" });
    expect(within(dialog).getByRole("button", { name: "Mark email verified" })).toBeInTheDocument();
    await u.click(within(dialog).getByRole("button", { name: "Make admin" }));
    await u.click(screen.getAllByRole("button", { name: "Make admin" }).at(-1)!);
    expect(await screen.findByText("You can't remove your own admin role.")).toBeInTheDocument();
  });

  it("shows health, runs service checks and the earnings refresh", async () => {
    const u = userEvent.setup();
    wrap(admin());
    await u.click(screen.getByRole("tab", { name: "Health" }));
    expect(await screen.findByText("All slots ran")).toBeInTheDocument();
    expect(screen.getByText("Morning digest")).toBeInTheDocument();
    expect(calls.some((c) => c.path === "/v1/admin/providers")).toBe(false);
    await u.click(screen.getByRole("button", { name: "Run checks" }));
    expect(await screen.findByText("Billing service")).toBeInTheDocument();
    await u.click(screen.getByRole("tab", { name: "Tools" }));
    await u.click(await screen.findByRole("button", { name: "Refresh earnings now" }));
    await waitFor(() => expect(screen.getByText(/dates found for 420 of 600 symbols/)).toBeInTheDocument());
  });
});
