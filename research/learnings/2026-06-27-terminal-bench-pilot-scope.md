# Terminal-Bench Pilot Scope

## Question

Should Terminal-Bench be part of the orchestration roadmap now, and if so, how
should it enter without distracting from the current BigCodeBench label loop?

## Decision

Add Terminal-Bench 2.1 as the first held-out agentic benchmark track after the
current BigCodeBench repeated-label expansion. Do not replace BigCodeBench with
Terminal-Bench.

BigCodeBench remains the cheaper source of clean worker-selection labels for
training the lightweight logits router. Terminal-Bench should test whether that
router generalizes to interactive terminal work where worker choice, tool use,
repair, verifier calls, and stop decisions matter.

## Artifact

Added `research/evals/terminal_bench_2p1_pilot_plan.json` and a metadata-only
pilot selector. The selector deliberately rejects prompt, instruction, oracle,
solution, and verifier fields so benchmark content does not leak into mempool
training corpora.

## Next Step

Finish the current held-out BigCodeBench active-policy diagnostic first. After
that, fetch or export Terminal-Bench task metadata only, select a 5-task pilot,
and run fixed single-worker baselines against the active orchestrator-selected
initial-worker policy.
