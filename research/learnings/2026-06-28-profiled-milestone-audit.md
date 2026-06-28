# Profiled Milestone Audit

## Question

Does the milestone audit reflect the latest profile-aware adaptive refresh
cycle and the existing Terminal-Bench pilot evidence?

## Result

Updated the milestone audit defaults to use current artifacts:

- adaptive refresh evidence:
  `research/refreshes/20260628-router-miss-repeat-24task-profiled-refresh-cycle.json`
- Terminal-Bench report:
  `research/evals/terminal_bench_2p1_fix_git_oracle_vs_qwen_next_report.json`

The adaptive refresh summary now exposes `promotion_profile`, so the milestone
status shows that the current refresh candidate was quarantined under the
`preserve_accuracy` gate.

Regenerated:

- `research/programs/milestone_status_20260628.json`
- `research/programs/milestones.json`

## Current Audit State

All milestone tracks have current evidence. The active milestone remains
`M6-adaptive-memory-refresh`, with this open gap:

- refresh cycle works but produced a quarantined candidate under
  `preserve_accuracy`

## Decision

Keep the audit update. It prevents stale Terminal-Bench paths and makes the
adaptive-refresh safety mode visible in roadmap-level status.
