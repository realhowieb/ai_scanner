import { afterEach, describe, expect, it, vi } from "vitest";

import { MARKER_KEY, readMarker, writeMarker } from "@/lib/lastVisit";

afterEach(() => { localStorage.clear(); vi.restoreAllMocks(); });

describe("last-visit marker", () => {
  it("reads seen and baseline, and nothing on a first visit", () => {
    expect(readMarker()).toEqual({});
    writeMarker("11:");
    expect(readMarker()).toEqual({ seen: 11 });
    writeMarker("12:11");
    expect(localStorage.getItem(MARKER_KEY)).toBe("12:11");
    expect(readMarker()).toEqual({ seen: 12, baseline: 11 });
    localStorage.setItem(MARKER_KEY, "junk:x");
    expect(readMarker()).toEqual({});
  });

  it("treats blocked storage as a first visit", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
    expect(readMarker()).toEqual({});
    expect(() => writeMarker("11:")).not.toThrow();
  });
});
