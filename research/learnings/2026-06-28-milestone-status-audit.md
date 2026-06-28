# Milestone Status Audit

Run tag: `20260628-milestone-status-audit`

The milestone ledger was stale: it still listed `M3-lightweight-router` as
active even though later loops had already produced the external benchmark
smoke/pilot evidence, active promoted lightweight policy, a local multi-head
orchestrator artifact, a Terminal-Bench pilot report, and an adaptive refresh
cycle.

This run added an artifact-backed audit report:

- `research/programs/milestone_status_20260628.json`
- `research/programs/milestones.json`

Current interpretation:

- M1-M4 are complete by their exit criteria.
- M5 is complete only in the research sense: the local multi-head orchestrator
  artifact exists and produced a clear negative result. It is quarantined, not
  promoted.
- M5.5 has a reproducible first Terminal-Bench comparison and trajectory
  evidence, but should remain a secondary harness until BigCodeBench routing is
  stronger.
- M6 has a working refresh cycle with privacy, evaluation, rollback, and
  promotion gates. The latest candidate was quarantined, so the active policy
  remains the 23-task lightweight logits router.

Next recommendation:

Keep M6 as the active improvement focus, but do not treat the original ladder as
finished. The next useful loop should improve the training signal that caused
the 50-task multi-head candidate to fail: more non-Qwen specialist rows,
latency-regret-aware features/objectives, and a stricter promotion gate for
solvable tasks.
