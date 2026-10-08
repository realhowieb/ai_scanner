import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Protected } from "@/components/Protected";
import { readCachedMe, writeCachedMe } from "@/session/meCache";
import { useSession } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/today",
}));

const fetchMock = vi.fn();
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

function Plan() {
  const { me: m } = useSession();
  return <p>plan {m?.plan}</p>;
}

describe("cached session", () => {
  it("renders the page at once from this tab's last /v1/me answer, then revalidates it", async () => {
    writeCachedMe(me("basic"));
    let answer: (r: Response) => void = () => {};
    fetchMock.mockImplementation(() => new Promise<Response>((resolve) => { answer = resolve; }));
    render(<Protected><Plan /></Protected>);
    expect(await screen.findByText("plan basic")).toBeInTheDocument();
    expect(String(fetchMock.mock.calls[0]?.[0]?.url ?? fetchMock.mock.calls[0]?.[0])).toContain("/v1/me");
    answer(jsonResponse(me("pro")));
    expect(await screen.findByText("plan pro")).toBeInTheDocument();
    expect(readCachedMe()?.plan).toBe("pro");
  });

  it("waits for /v1/me when nothing is cached, and drops the cache when the session has expired", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "expired", code: "session_expired" }, 401));
    const assign = vi.fn();
    vi.stubGlobal("location", { ...window.location, assign, pathname: "/today", search: "" });
    render(<Protected><Plan /></Protected>);
    expect(screen.queryByText(/^plan/)).not.toBeInTheDocument();
    await waitFor(() => expect(assign).toHaveBeenCalled());
    expect(readCachedMe()).toBeNull();
  });
});
