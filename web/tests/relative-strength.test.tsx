import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { VsSpy } from "@/features/SetupTable";

describe("vs SPY cell", () => {
  it("shows the 20-session RS with sign and the RS high badge", () => {
    render(<VsSpy rs={8.5} high />);
    expect(screen.getByText("+8.50%")).toHaveClass("up");
    expect(screen.getByText("RS high")).toBeInTheDocument();
  });

  it("shows a lagging stock without a badge", () => {
    render(<VsSpy rs={-2} high={false} />);
    expect(screen.getByText("-2.00%")).toHaveClass("down");
    expect(screen.queryByText("RS high")).toBeNull();
  });

  it("shows a dash without a number", () => {
    render(<VsSpy rs={null} high={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
