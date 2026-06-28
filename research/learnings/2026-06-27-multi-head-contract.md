# Multi-Head Orchestrator Contract

Added a bridge contract for the small trainable orchestrator:

- `src/mempool/orchestrator_contract.py`
- `tools/export_multi_head_orchestrator_contract.py`
- `research/models/20260627-active-multi-head-orchestrator-contract.json`

The contract defines four required action heads:

- `worker_distribution`: softmax head copied from the active logits router
- `workflow_kind`: softmax bridge prior over `direct` and `verify_then_fallback`
- `verifier_probability`: sigmoid head seeded from the selected gated fallback
  teacher
- `abstain_probability`: sigmoid bridge prior for future high-risk or unsolved
  labels

The readiness audit now validates this contract. After regenerating
`research/programs/small_orchestrator_readiness.json`, the workflow-kind and
abstain blockers are cleared. The only remaining M5 blocker is data volume:
the active repeated routing dataset has 23 tasks and the threshold is 50.

Next step: collect stable repeated BigCodeBench rows, preferably specialist-win
or latency-tie rows, until the active dataset clears the 50-task gate.
