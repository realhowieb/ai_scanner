import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Schemas } from "@/api/client";
import { TodayView } from "@/features/TodayView";

import { greeting, localDate } from "@/lib/format";

// 15:00 UTC on Wed Oct 21 2026: 11 AM in New York, 8 AM in Los Angeles, 4 PM in London, 1 AM Thu in Tokyo.
const T = Date.parse("2026-10-21T15:00:00Z");

describe("Local greeting", () => {
  it("follows the viewer's time zone, not ET", () => {
    expect(greeting(T, "America/New_York")).toBe("Good morning");
    expect(greeting(T, "America/Los_Angeles")).toBe("Good morning");
    expect(greeting(T, "Europe/London")).toBe("Good afternoon");
    expect(greeting(Date.parse("2026-10-21T21:30:00Z"), "America/Chicago")).toBe("Good afternoon");
    expect(greeting(Date.parse("2026-10-21T23:30:00Z"), "America/Chicago")).toBe("Good evening");
  });

  it("shows midnight as morning and noon as afternoon", () => {
    expect(greeting(Date.parse("2026-10-21T04:00:00Z"), "America/New_York")).toBe("Good morning");
    expect(greeting(Date.parse("2026-10-21T16:00:00Z"), "America/New_York")).toBe("Good afternoon");
  });

  it("uses the viewer's calendar date", () => {
    expect(localDate(T, "America/New_York")).toBe("Wednesday, October 21");
    expect(localDate(T, "Asia/Tokyo")).toBe("Thursday, October 22");
  });

  it("defaults to the browser time zone", () => {
    expect(greeting(T)).toBe(greeting(T, Intl.DateTimeFormat().resolvedOptions().timeZone));
  });

  it("renders on Today from the browser clock, not the scan time", () => {
    // Scan saved at 11 PM ET yesterday; the page should still greet by the viewer's current time.
    const data = { as_of: "2026-10-20T03:00:00Z", market: { phase: "closed" }, errors: [], before_open: null, top_setups: null, after_close: null, recap: null } as unknown as Schemas["Today"];
    render(<TodayView data={data} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(greeting(Date.now()));
    expect(screen.getByText(localDate(Date.now()))).toBeInTheDocument();
  });
});
