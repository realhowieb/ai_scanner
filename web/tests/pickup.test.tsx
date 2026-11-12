import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PickedCell, PickupLine, pickupWhen, sincePickup } from "@/features/Pickup";

// Thu Nov 12 2026, 2:00 PM ET
const NOW = Date.parse("2026-11-12T19:00:00Z");
const pickup = { at: "2026-11-12T14:35:00+00:00", price: 100, rank: 30, open_ended: false };

describe("pickup", () => {
  it("names the time in ET: today, this week, earlier", () => {
    expect(pickupWhen("2026-11-12T14:35:00Z", NOW)).toBe("9:35 AM ET");
    expect(pickupWhen("2026-11-09T17:35:00Z", NOW)).toBe("Mon 12:35 PM ET");
    expect(pickupWhen("2026-11-02T17:35:00Z", NOW)).toBe("Nov 2, 12:35 PM ET");
    expect(pickupWhen(null, NOW)).toBe("");
  });

  it("measures the move since the pickup price", () => {
    expect(sincePickup(pickup, 103)).toBeCloseTo(3);
    expect(sincePickup({ ...pickup, price: null }, 103)).toBeNull();
    expect(sincePickup(pickup, null)).toBeNull();
  });

  it("shows the time, price then and move in the Scanner cell", () => {
    render(<PickedCell pickup={pickup} last={103} nowMs={NOW} />);
    expect(screen.getByText("9:35 AM ET")).toBeTruthy();
    expect(screen.getByText("+3.0%")).toBeTruthy();
  });

  it("writes one sentence for the quick view and stock page", () => {
    const { container, rerender } = render(<PickupLine pickup={pickup} last={103} nowMs={NOW} />);
    expect(container.textContent).toBe("Picked up by HSF at 9:35 AM ET at $100.00 (#30 then) · +3.0% since");
    rerender(<PickupLine pickup={{ ...pickup, open_ended: true }} last={100} nowMs={NOW} />);
    expect(container.textContent).toBe("In every HSF scan since at least 9:35 AM ET at $100.00 (#30 then)");
    rerender(<PickupLine pickup={null} last={100} nowMs={NOW} />);
    expect(container.textContent).toBe("");
  });
});
