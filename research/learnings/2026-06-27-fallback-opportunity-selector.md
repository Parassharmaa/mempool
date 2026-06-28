# Fallback Opportunity Acquisition Selector

## Question

After the fallback logit head failed to calibrate cleanly, can we select the
next BigCodeBench tasks specifically to create useful fallback/verifier labels?

## Change

Added `tools/select_fallback_opportunity_batch.py`.

The selector ranks fresh BigCodeBench tasks by:

- low top-two probability margin from the active logits router
- similarity to known fallback-relevant seeds
- preferred second workers such as GLM and DeepSeek
- lower environment risk

It writes a task file for the next real worker run and an auditable report:

- `research/evals/bigcodebench_hard_fallback_opportunity_batch8_tasks.json`
- `research/evals/bigcodebench_hard_fallback_opportunity_batch8_report.json`

## Result

From the merged 69-task eligible pool, after excluding already routed or mined
tasks, the selector considered 32 fresh candidates and selected 8:

- `BigCodeBench/771`
- `BigCodeBench/785`
- `BigCodeBench/509`
- `BigCodeBench/857`
- `BigCodeBench/15`
- `BigCodeBench/346`
- `BigCodeBench/486`
- `BigCodeBench/988`

The top selected candidates are mostly filesystem/subprocess tasks where the
active router is uncertain:

- `BigCodeBench/771`: margin 0.0441, Qwen top, Kimi second
- `BigCodeBench/785`: margin 0.0576, Qwen top, Kimi second
- `BigCodeBench/509`: margin 0.0661, Qwen top, Kimi second
- `BigCodeBench/857`: DeepSeek second, selected despite wider margin because it
  is close to the specialist/fallback acquisition target

## Learning

The next fallback dataset should not be a generic fresh batch. It should target
first-worker failure opportunities where the router already has uncertainty and
where the second-ranked worker is a plausible specialist. This gives the
fallback head a chance to learn a calibrated action boundary instead of
overfitting one or two rescue examples.

This is an acquisition plan, not outcome evidence yet. The selected tasks need
real repeated worker runs before they can be added to routing or fallback-head
training.

## Next Step

Run the selected batch against the current cloud worker pool with repeated
samples, then convert it into a fallback-opportunity training slice containing:

- top worker pass/fail
- second worker pass/fail
- whether fallback improved pass rate
- extra latency paid by fallback
- whether GLM, DeepSeek, or Kimi was the true rescue worker
