# Router Miss Repeat Batch

## Question

Can the router-miss acquisition planner produce a fresh task batch near the
current leave-one-out miss regions?

## Result

The strict fresh candidate pool is exhausted under the current BigCodeBench-Hard
task materializations. All tasks in the known eligible/fresh pools are already
present in routing datasets or outcome JSONL files.

A dependency-profile expansion was also planned, but the next four packages
would only unlock four niche/heavy tasks:

- `geopandas`
- `soundfile`
- `regex`
- `tensorflow`

That is not the best immediate spend for router improvement.

## Decision

Keep a controlled fallback in the miss-neighborhood selector: when no fresh
candidates are available, it can materialize the miss seed tasks themselves as a
repeat-stabilization batch. This is explicitly marked with
`selection_reason: fallback_seed_repeat`.

Generated artifacts:

- `research/evals/20260628-router-miss-neighborhood-repeat4-report.json`
- `research/evals/20260628-router-miss-neighborhood-repeat4-tasks.json`
- `research/evals/20260628-router-miss-repeat4-top4-manifest.json`
- `research/evals/20260628-router-miss-next-dependency-profile-plan.json`

The repeat manifest is ready for a live top4 run:

- 4 miss seed tasks
- 4 cloud workers
- 2 repeats
- 32 total worker calls

## Next Step

Run the manifest once `OLLAMA_API_KEY` is loaded into the shell environment.
Then summarize outcomes, rebuild the routing rows, and re-run the gated policy
refresh. Promote only if leave-one-out target accuracy and latency regret beat
the active 23-task baseline.
