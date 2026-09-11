"""Smoke-test the 'Why this AI Confidence' explainer with real Claude output.

Needs ANTHROPIC_API_KEY set. Builds a sample scanner row (no scan/DB/admin
needed) and prints Claude's explanation. Usage:

    ANTHROPIC_API_KEY=sk-ant-... python scripts/try_confidence_explain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from ui.ai_confidence_explain import confidence_tier, explain_confidence

    # A representative "high confidence" row: strong RVOL, improving trend,
    # sitting just under a breakout, with earnings 2 days out.
    row = {
        "Ticker": "AMD",
        "AI Confidence": 45.0,
        "Trend10D%": 3.1,
        "Trend20D%": 2.2,
        "VolRel20": 2.4,
        "DollarVol20": 1.8e9,
        "BreakoutScore": 7.0,
        "GapPct": 1.2,
    }
    print(f"Ticker: {row['Ticker']}  |  Confidence: {row['AI Confidence']}%  "
          f"(tier: {confidence_tier(row['AI Confidence'])})")
    print("Asking Claude…\n")
    text, err = explain_confidence(row, earnings_days=2)
    if err:
        print(f"ERROR: {err}")
        print("(Set ANTHROPIC_API_KEY and ensure AI_ENABLED != 0.)")
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
