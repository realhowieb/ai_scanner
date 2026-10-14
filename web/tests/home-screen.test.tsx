import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import manifest from "@/app/manifest";
import { InstallHint } from "@/components/InstallHint";

const IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1";

function asDevice(ua: string, standalone = false) {
  vi.spyOn(navigator, "userAgent", "get").mockReturnValue(ua);
  Object.defineProperty(navigator, "standalone", { value: standalone, configurable: true });
}

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
  delete (navigator as { standalone?: boolean }).standalone;
});

describe("Home Screen app", () => {
  it("gives iPhones PNG icons, since they ignore SVG", () => {
    const icons = manifest().icons ?? [];
    expect(icons.some((i) => i.type === "image/png" && i.sizes === "512x512")).toBe(true);
    expect(icons.some((i) => i.type === "image/png" && i.sizes === "192x192")).toBe(true);
  });

  it("tells iPhone Safari users how to add HSF, and remembers Not now", async () => {
    asDevice(IPHONE);
    const { unmount } = render(<InstallHint />);
    expect(await screen.findByText(/Add to Home Screen/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Not now" }));
    expect(screen.queryByText(/Add to Home Screen/)).not.toBeInTheDocument();
    unmount();
    render(<InstallHint />);
    expect(screen.queryByText(/Add to Home Screen/)).not.toBeInTheDocument();
  });

  it("stays hidden in the installed app and on desktop", () => {
    asDevice(IPHONE, true);
    const { unmount } = render(<InstallHint />);
    expect(screen.queryByText(/Add to Home Screen/)).not.toBeInTheDocument();
    unmount();
    asDevice("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15");
    render(<InstallHint />);
    expect(screen.queryByText(/Add to Home Screen/)).not.toBeInTheDocument();
  });
});
