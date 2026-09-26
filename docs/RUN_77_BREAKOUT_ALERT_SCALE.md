# Run 77 - Breakout Alert Threshold on a Visible Scale

## Problem

The Breakout alert form previously said only “Breakout score >=” and defaulted
to `8.0`. Users could not tell whether this meant HSF Score, what direction made
an alert more selective, or how the value related to recent scanner output.

The underlying behavior was already correct and remains unchanged:

```text
latest scan BreakoutScore >= saved threshold
```

## Decision

Keep the existing Breakout Score alert rather than silently changing it to HSF
Score. Changing the evaluated score would alter alert semantics, existing saved
alerts, firing frequency, and historical comparability.

The UI now states permanently that:

- the threshold uses **Breakout Score**, a supporting technical scanner score;
- it is **not the 0-100 HSF Score**;
- lower values are more frequent and higher values are more selective;
- the comparison is `Breakout Score >= threshold`;
- `8.0` remains the existing default.

Breakout Score is additive and is not presented as a fixed 0-100 scale. The
existing opt-in threshold-history view is renamed to “Show threshold history
and observed scale.” It plots recent daily top Breakout Scores and the selected
threshold, providing an honest empirical scale rather than an invented bound.

Saved-alert summaries now say “Breakout Score >= N” instead of the ambiguous
“Breakout >= N.”

## Safety

No changes were made to:

- `scheduler/alert_runner.py` evaluation;
- alert database schema or saved values;
- default threshold, minimum, or step;
- HSF Score, Breakout Score, scanner logic, ranking, models, or research;
- alert delivery, dedupe, throttling, or preferences.

Existing alerts retain their exact meaning and continue to fire under the same
condition.

## Tests

Coverage verifies the visible explanation and legend, default threshold,
unambiguous saved-alert label, preserved runner comparison, empirical preview
wiring, and separation between UI, storage, and evaluation logic.

P1-12 can be marked **DONE**.
