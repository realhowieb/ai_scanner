import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { OutcomeScores } from "@/api/account";
import { ScoreOrderNote } from "@/features/OutcomeEvidence";

function scores(inversions: Array<{ lower: string; higher: string; holds: boolean }>): OutcomeScores {
  return {
    calibration: {
      min_matured_per_bucket: 10,
      metrics: {
        median_excess_return: {
          buckets_compared: ["50-59", "60-69", "70-79"],
          monotonic: inversions.length === 0,
          inversions: inversions.map((i) => ({
            lower_bucket: i.lower, higher_bucket: i.higher, lower_value: 0.02, higher_value: 0,
            gap_ci95: { low: i.holds ? 0.01 : -0.01, high: 0.03 }, holds_across_days: i.holds,
          })),
        },
      },
    },
  } as unknown as OutcomeScores;
}

describe("By HSF score footnote", () => {
  it("says Not monotonic only for gaps that hold across days", () => {
    render(<ScoreOrderNote d={scores([{ lower: "50-59", higher: "60-69", holds: true }, { lower: "60-69", higher: "70-79", holds: false }])} />);
    expect(screen.getByText("Not monotonic")).toBeInTheDocument();
    expect(screen.getByText(/HSF 50-59 beat 60-69 vs SPY, and the gap holds across days/)).toBeInTheDocument();
    expect(screen.queryByText(/60-69 beat 70-79/)).toBeNull();
  });

  it("says No clear difference yet when no gap holds", () => {
    render(<ScoreOrderNote d={scores([{ lower: "0-49", higher: "50-59", holds: false }])} />);
    expect(screen.queryByText("Not monotonic")).toBeNull();
    expect(screen.getByText("No clear difference yet")).toBeInTheDocument();
    expect(screen.getByText(/0-49 over 50-59/)).toBeInTheDocument();
  });

  it("shows nothing when the ranges are in order", () => {
    const { container } = render(<ScoreOrderNote d={scores([])} />);
    expect(container).toBeEmptyDOMElement();
  });
});
