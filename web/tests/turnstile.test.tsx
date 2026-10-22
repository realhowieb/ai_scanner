import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
  delete (window as { turnstile?: unknown }).turnstile;
});

describe("Turnstile widget", () => {
  it("shows Cloudflare's error code instead of failing silently", async () => {
    vi.stubEnv("NEXT_PUBLIC_TURNSTILE_SITE_KEY", "0xTEST");
    let opts: Record<string, (...a: unknown[]) => void> = {};
    (window as { turnstile?: unknown }).turnstile = {
      render: (_el: HTMLElement, o: Record<string, (...a: unknown[]) => void>) => { opts = o; return "w1"; },
      reset: vi.fn(), remove: vi.fn(),
    };
    const { Turnstile } = await import("@/components/Turnstile");
    const onToken = vi.fn();
    render(<Turnstile action="password_reset" onToken={onToken} />);
    await waitFor(() => expect(opts["error-callback"]).toBeTypeOf("function"));
    act(() => opts["error-callback"]!("110200"));
    expect(await screen.findByRole("alert")).toHaveTextContent("Cloudflare error 110200");
    expect(onToken).toHaveBeenLastCalledWith(null);
    act(() => opts.callback!("tok"));
    expect(onToken).toHaveBeenLastCalledWith("tok");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
