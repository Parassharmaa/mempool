# Terminal-Bench Readiness Gate

Terminal-Bench 2.1 remains a good held-out agentic benchmark candidate, but the
local Harbor path is not ready for worker comparisons yet.

The readiness gate now requires a safe preflight summary with:

- `process_status: exited`
- `harbor_summary.status: complete`

Current summaries do not satisfy that contract. The easiest install-only
preflight timed out after 300 seconds with `harbor_summary.status:
running_or_stale`, while earlier manual summaries were produced before the
wrapper recorded process-level status. Keep Terminal-Bench behind this gate
until a reproducible clean preflight completes.
