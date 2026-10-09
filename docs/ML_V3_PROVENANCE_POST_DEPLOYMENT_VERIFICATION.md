# ML v3 Post-Deployment Provenance Verification

## Initial findings (before fixes)

On inspection, latest main is `e62b229500ccb4873fab1c26d4f08d2bd2e328c1`.
Latest successful Scheduled Market Scans execution is `37960183886`, created
2026-10-09 16:35:13 UTC, running prior SHA
`4866e642ff4f06fba731ff5ce71c3652cc1c54b1`.
No post-merge scheduled execution was available at the first check.
Execution status: **WAITING_FOR_PROSPECTIVE_RUN**, not deployment verified.

Source findings:
1. `ui.results.render_results` permits ModelProvenance into customer tables,
   raw previews and CSV exports; it contains internal AI probabilities and inputs.
2. `cron_runner._results_to_json` uses pandas default numeric precision.
   Nested calibration snapshots round while their hashes retain original
   precision. Final feature vectors can also lose precision.
3. Scheduled execution calls `scan.engine.run_breakout_scan` directly, not
   `scan.execution.run_manual_scan_execution`. The engine does not invoke
   AI-confidence inference. PreBreakout is explicitly invoked afterward.
   Therefore scheduled AI provenance is unavailable by architecture, not a
   serialization failure. Adding AI inference would change scheduled behavior
   and is intentionally outside this verification.

Narrow proposed fixes: remove internal columns from the rendering copy only;
serialize provenance as a lossless JSON string inside the DataFrame (existing
models_from_row already accepts strings), then decode at observation/freezing
boundaries. No inference or scoring change is needed.
