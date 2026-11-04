# API memory follow-up — November 4, 2026

The reported peak was 414.3 MB on a 512 MB instance. Two subsequent live readiness samples showed roughly 459.4–459.7 MB current container memory (452.7 MB process RSS). This was current pressure, not just an old peak. The API remained healthy. The price cache contained two entries.

Render runtime logs were unavailable through the connected tools. The investigation therefore does not establish which operation produced the production increase or how much each change will recover.

Code inspection found that expired shared-cache values could remain until another write and failed loaders left per-key coordination locks behind. Cache reads and five-minute operations maintenance now remove entries past their allowed stale window. Foreground and background loaders share reference-counted coordination locks, including failure cleanup. Valid cached values remain available.

Under current pressure (HSF_MEMORY_WARN_MB, default 400 MB), maintenance collects garbage and asks glibc to return unused allocator pages. It does not restart the service, change the instance plan, clear valid cached values, or reset recorded peaks. HSF_MEMORY_MAINTENANCE_ENABLED=0 disables maintenance. Other platforms skip the allocator step.

Readiness and health evidence now include aggregate cache counts. Logs identify RSS increases of at least 16 MB around named cache builders and custom scans. Concurrent work can contribute to these process-wide deltas; they are investigation clues, not allocation attribution. Maintenance logs show before/after memory and expired-entry counts. Cache keys and user identities are excluded.

After deployment, compare current container memory and maintenance logs across normal scan and page traffic. Peaks remain historical; verify current usage rather than expecting the peak warning to clear immediately.

# Completed evaluation notice

Health collection reads workflow metadata, not effectiveness results. A successful, completed, manually dispatched main-branch Signal Effectiveness Analysis created within the declared forward epoch replaces the repeated dispatch recommendation with a completed-run review notice. The epoch's baseline run is excluded. Failed, scheduled, development-branch, pre-epoch and future runs do not qualify. Data-quality blocks retain priority.

The admin research panel links to the completed run. Completion does not imply a positive edge, authorize another evaluation, or change ML training readiness. No evaluation is dispatched automatically.
