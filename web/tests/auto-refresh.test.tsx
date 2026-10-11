import { act, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { inMarketHours, useAutoRefresh } from "@/hooks/useAutoRefresh";

function Probe({ reload }: { reload: () => void }) {
  useAutoRefresh(reload, { everyMs: 60_000 });
  return null;
}

describe("auto-refresh", () => {
  afterEach(() => vi.useRealTimers());

  it("knows US market hours in New York time", () => {
    expect(inMarketHours(new Date("2026-10-13T14:00:00Z"))).toBe(true);   // Tue 10:00 ET
    expect(inMarketHours(new Date("2026-10-13T12:00:00Z"))).toBe(false);  // Tue 8:00 ET
    expect(inMarketHours(new Date("2026-10-13T21:00:00Z"))).toBe(false);  // Tue 17:00 ET
    expect(inMarketHours(new Date("2026-10-11T15:00:00Z"))).toBe(false);  // Sunday
  });

  it("reloads on the interval during market hours only", () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "Date"] });
    vi.setSystemTime(new Date("2026-10-13T14:00:00Z"));
    const reload = vi.fn();
    const { unmount } = render(<Probe reload={reload} />);
    act(() => vi.advanceTimersByTime(60_000));
    expect(reload).toHaveBeenCalledTimes(1);
    vi.setSystemTime(new Date("2026-10-13T22:00:00Z"));
    act(() => vi.advanceTimersByTime(180_000));
    expect(reload).toHaveBeenCalledTimes(1);
    unmount();
  });
});
