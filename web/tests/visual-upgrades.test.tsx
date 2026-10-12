import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ScoreLadder } from "@/features/OutcomeEvidence";
import { ScoreHistory } from "@/features/ScoreHistory";
import { DailySpark, RvolMeter } from "@/features/Sparks";

import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("Row sparklines and RVOL meter", () => {
  it("draws 20 sessions, coloured by the move, and reads the change aloud", () => {
    render(<DailySpark points={[10, 11, 9, 12]} ticker="AAA" />);
    const svg = screen.getByRole("img", { name: /AAA last 3 sessions: up 20\.0%/ });
    expect(svg).toHaveClass("spark", "up");
  });

  it("shows a dash without enough closes", () => {
    const { container } = render(<DailySpark points={[10]} ticker="AAA" />);
    expect(container).toHaveTextContent("—");
  });

  it("fills the meter toward 3x and turns amber from 1.5x", () => {
    const { container } = render(<RvolMeter rvol={1.8} />);
    const fill = container.querySelector(".rvol-fill") as HTMLElement;
    expect(fill).toHaveClass("hot");
    expect(fill.style.width).toBe("60%");
  });
});

describe("Score ladder", () => {
  it("puts every range on one zero line with its sample size, fading thin ones", () => {
    const b = (bucket: string, v: number, n: number, q = "MODERATE") =>
      ({ bucket, median_excess_return: v, matured_count: n, evidence_quality: q }) as never;
    const { container } = render(<ScoreLadder buckets={[b("0-49", -0.0194, 247), b("90-100", 0.0107, 2, "INSUFFICIENT")]} />);
    expect(screen.getByText("n=247")).toBeInTheDocument();
    expect(screen.getByText("-1.94%")).toHaveClass("down");
    expect(container.querySelectorAll(".ladder-thin")).toHaveLength(1);
  });
});

describe("HSF Score history", () => {
  it("charts the Scanner's score per scan, with gaps and signal marks", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ticker: "AAA", points: [
      { time: "2026-10-08T13:35:00Z", score: 50, signals: ["breakout"], signals_added: [] },
      { time: "2026-10-09T13:35:00Z", score: null, signals: [], signals_added: [] },
      { time: "2026-10-12T19:35:00Z", score: 77, signals: ["breakout", "golden_cross"], signals_added: ["golden_cross"] },
    ] }));
    const { container } = render(<ScoreHistory ticker="AAA" />);
    expect(await screen.findByRole("img", { name: /50 to 77, ranked in 2 of 3/ })).toBeInTheDocument();
    expect(container.querySelectorAll(".sh-line")).toHaveLength(2);   // the unranked scan breaks the line
    expect(container.querySelectorAll(".sh-signal")).toHaveLength(1);
    expect(new URL((fetchMock.mock.calls[0]![0] as Request).url).pathname).toMatch(/\/v1\/stocks\/AAA\/score-history$/);
  });
});
