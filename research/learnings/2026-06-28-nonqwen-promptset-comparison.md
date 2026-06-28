# Non-Qwen Prompt-Set Comparison

Run tag: `20260628-nonqwen-promptset-comparison`

This run selected prompts directly from measured non-Qwen target regions in the
66-record task-level substrate, then compared the trained orchestrator against a
fixed `ollama-cloud-qwen3-coder-480b` baseline.

## Prompt Set

Generated with `tools/select_prompt_set_from_substrate.py`.

Artifacts:

- Prompt set: `research/evals/20260628-nonqwen-promptset-comparison-prompts.json`
- Full comparison: `research/evals/20260628-nonqwen-promptset-comparison.json`
- Outcome rows: `research/evals/20260628-nonqwen-promptset-comparison-outcomes.jsonl`

Selected target regions:

- `BigCodeBench/339`: target `ollama-cloud-glm-5.2`
- `BigCodeBench/1004`: target `ollama-cloud-kimi-k2.7-code`
- `BigCodeBench/1053`: target `ollama-cloud-deepseek-v4-pro`

## Result

The trained orchestrator selected a different non-Qwen worker for every prompt:

- `BigCodeBench/339`: selected `ollama-cloud-glm-5.2`
- `BigCodeBench/1004`: selected `ollama-cloud-kimi-k2.7-code`
- `BigCodeBench/1053`: selected `ollama-cloud-deepseek-v4-pro`

The fixed baseline used `ollama-cloud-qwen3-coder-480b` for all prompts.

Mean latency:

- trained orchestrator policy: 6955.0 ms
- fixed Qwen baseline: 4052.0 ms

Responses were present for all six live executions.

## Decision

Keep this run as the first live proof that the trained checkpoint can route away
from Qwen when prompts come from measured non-Qwen regions. Do not promote it as
a latency win: all three orchestrator-selected non-Qwen calls were slower than
the fixed Qwen baseline on this live sample. The next refinement should compare
execution quality with an evaluator or use latency-aware policy calibration on
these non-Qwen regions.
