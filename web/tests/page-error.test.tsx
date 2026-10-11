import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PageError } from "@/components/PageError";
import { reportError, resetReportedErrors } from "@/lib/reportError";

describe("page error screen", () => {
  const fetchMock = vi.fn(() => Promise.resolve(new Response(null, { status: 202 })));
  beforeEach(() => {
    resetReportedErrors();
    fetchMock.mockClear();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("says what happened, offers Try again and reports the crash once", () => {
    const retry = vi.fn();
    const error = Object.assign(new Error("cannot read properties of undefined"), { digest: "abc123" });
    render(<PageError error={error} retry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong on this page");
    expect(screen.getByText("Support code: abc123")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/public/v1/client-errors");
    const sent = JSON.parse(String(init.body));
    expect(sent).toMatchObject({ message: "cannot read properties of undefined", kind: "boundary", digest: "abc123", path: "/" });
    expect(init.credentials).toBe("omit");
  });

  it("sends the same message once and at most five per page load", () => {
    reportError(new Error("same"));
    reportError(new Error("same"));
    for (let i = 0; i < 10; i++) reportError(new Error(`e${i}`));
    expect(fetchMock).toHaveBeenCalledTimes(5);
  });
});
