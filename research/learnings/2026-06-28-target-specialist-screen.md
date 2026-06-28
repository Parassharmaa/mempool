# Target Specialist Screen

Run tag: `20260628-glm-target-screen`

Goal: test whether specialist-acquisition candidates should be screened by the
intended specialist worker before graduating to full repeated four-worker
comparison.

Change:

Added target-worker filtering to `tools/plan_acquisition_batch.py` and added
`tools/filter_worker_pool.py` so the acquisition loop can produce a one-worker,
one-sample screen for a specific specialist cluster.

Screens run:

GLM target screen:

- Worker: `ollama-cloud-glm-5.2`
- Tasks: `BigCodeBench/528`, `BigCodeBench/857`
- Calls: `2`
- Passes: `0`
- Graduated tasks: `0`

Kimi target screen:

- Worker: `ollama-cloud-kimi-k2.7-code`
- Tasks: `BigCodeBench/211`, `BigCodeBench/1129`
- Calls: `2`
- Passes: `0`
- Graduated tasks: `0`

Decision: keep the target-specialist screen machinery.

The data result is negative: these target screens did not produce training rows.
But the workflow result is useful because it prevents wasting full repeated
comparison runs on candidates that the intended specialist cannot solve even
once.

Current acquisition implication:

The first specialist-acquisition queue is now weak. We have screened or
full-compared a large fraction of it, and every attempted candidate has failed
to produce a passing row. The next acquisition step should generate fresh
candidate tasks using a stronger solvability prior, not continue repeating the
same miss-neighborhood queue.

Next step:

Create a fresh acquisition selector that starts from tasks with canonical
environment confidence or historical positive signals, then uses specialist
similarity only after the candidate is likely solvable. Keep the
target-specialist screen as the graduation gate.

Artifacts:

- `tools/plan_acquisition_batch.py`
- `tools/filter_worker_pool.py`
- `research/evals/ollama_cloud_glm_single_pool.json`
- `research/evals/ollama_cloud_kimi_single_pool.json`
- `research/evals/ollama_cloud_deepseek_v4_pro_single_pool.json`
- `research/programs/20260628-specialist-acquisition-glm-screen1.json`
- `research/programs/20260628-specialist-acquisition-kimi-screen1.json`
- `research/evals/results/20260628-specialist-acquisition-glm-screen1-summary.json`
- `research/evals/results/20260628-specialist-acquisition-kimi-screen1-summary.json`
