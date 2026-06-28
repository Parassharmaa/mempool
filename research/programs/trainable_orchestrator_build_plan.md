# Trainable Orchestrator Build Plan

This is the long-term execution plan for building `mempool` into a trainable
local orchestration system.

The plan should be executed through the research loop. Each milestone must leave
behind:

- code or data artifact
- evaluation result
- learning note
- next-step recommendation

## End Goal

Build a small, frequently updatable orchestrator that can route tasks across a
mixed local/cloud worker pool, learn from measured outcomes, and gradually fuse
useful memory into lightweight policy updates.

## Milestone 1: Real Worker Pool Evaluation

Goal: replace fixture-only smoke signals with real Ollama/OpenAI-compatible
worker calls.

Tasks:

- define a worker-pool config for selected Ollama models
- run the local smoke benchmark against each worker
- record output, pass/fail, cost estimate, latency, and failure mode
- compare cheapest, strongest, and rule-routed baselines

Exit criteria:

- at least 3 real workers evaluated on the local smoke set
- JSONL outcome records produced
- first real cost/pass/latency comparison documented

## Milestone 2: Outcome Dataset

Goal: convert benchmark runs into canonical training records.

Tasks:

- define the training-record schema
- write a converter from benchmark results to JSONL
- compute per-worker rewards
- generate soft routing targets from worker rewards

Exit criteria:

- dataset file with task features, worker outcomes, and routing targets
- dataset validation tests
- learning note on whether task features separate worker performance

## Milestone 3: Lightweight Router

Goal: train the first non-LLM routing policy.

Tasks:

- implement a simple feature extractor
- train a baseline classifier/ranker
- compare learned router against rule router, cheapest worker, and strongest
  worker
- log cost-per-solved-task and degradation cases

Exit criteria:

- trained router artifact
- evaluation report against baselines
- decision on whether to proceed to neural/LLM orchestrator

## Milestone 4: External Smoke Benchmark

Goal: move beyond toy local tasks to a 10-task external code benchmark smoke set.

Tasks:

- add BigCodeBench-Hard smoke loader
- run selected workers on 10 tasks
- evaluate generated code reproducibly
- update router dataset with external outcomes

Exit criteria:

- 10-task benchmark report
- comparison of rule router vs learned router vs strongest worker
- documented failure modes

## Milestone 5: Small Trainable Orchestrator

Goal: train a small neural or language-model orchestrator that emits action
logits directly.

Tasks:

- choose base path: embedding plus MLP, sub-1B hidden-state backbone, or 1-2B
  LoRA
- train on measured routing data
- output logits for worker distribution, workflow kind, verifier probability,
  and abstain probability
- train against soft worker targets with KL divergence before any rollout RL
- evaluate against prior router

Exit criteria:

- local trainable logits-head orchestrator artifact
- measurable improvement or clear negative result
- documented compute, latency, and update cost

## Milestone 6: Adaptive Memory Refresh

Goal: begin converting repeated experience into model updates.

Tasks:

- add trace sanitization and distillation pipeline
- version distilled datasets
- run policy refresh on a cadence
- evaluate and rollback failed updates

Exit criteria:

- end-to-end ledger-to-training-record path
- at least one evaluated refresh cycle
- guardrails documented for privacy and rollback

## Milestone 5.5: Agentic Harness Pilot

Goal: evaluate the orchestrator in an interactive terminal harness before
adaptive memory refresh.

Tasks:

- add a Terminal-Bench 2.1 adapter or wrapper
- run a tiny subset with cloud workers and local workers
- record terminal actions, tool calls, file edits, tests, reward, cost, and
  latency
- compare single-worker agents against orchestrator-selected workers

Exit criteria:

- at least one reproducible Terminal-Bench subset report
- ledger records for multi-turn terminal trajectories
- decision on whether the logits-head router needs state/history features

## Current Priority

Milestones 1 and 2 are complete. Milestone 3 is active. Expanding the smoke set
from 4 to 10 tasks improved evidence quality, but leave-one-out routing still
does not beat strongest-worker pass rate. Add more evaluated tasks and improve
the run harness before moving to a neural router.

The real-worker runner now supports resumable evaluation with progress logging,
which makes longer 30-50 task pilots and external smoke benchmarks practical.
Next, add the external benchmark loader and keep using resumable per-worker
writes while collecting benchmark-backed outcomes.

The external BigCodeBench-Hard loader is now in place, and the first one-task
Ollama probe produced a valid benchmark-backed failure signal across all three
workers. Before scaling to the full 10-task smoke set, add a selector for a
3-task mini-pilot so the next external run includes varied and plausibly
solvable tasks instead of only universal failures.

The mini-pilot selector now filters tasks by canonical-solution pass status in
the current local environment. Only three of the first ten materialized hard
tasks are currently eligible, so the next run should evaluate that 3-task file
with resumable worker calls and then decide whether to install a broader
benchmark dependency profile.

The 3-task mini-pilot has been evaluated across the current Ollama worker pool.
It produced one solved task and two universal failures. This is enough to prove
the external outcome-to-routing-data path, but not enough to train a neural
orchestrator. Next priority is expanding benchmark-eligible external data,
either by selecting more standard-library tasks or by adding an isolated
benchmark dependency profile.

The eligible-pool scan found 8 canonical-pass BigCodeBench-Hard tasks after
probing 44 rows in the current local environment. Use this 8-task pool for the
next external worker run, unless we first decide to add an isolated dependency
profile for data-science and web tasks.

The 8-task pool has now been evaluated. It yielded only one positive task,
solved by both Qwen workers, with qwen3-4b-instruct much faster than qwen3-1.7b.
Do not move to neural orchestrator training yet. Expand positive external
outcomes first by continuing the standard-library scan or adding an isolated
benchmark dependency profile.

Architecture update: the small trainable orchestrator should be a
logits-emitting decision model. It should use a compact backbone representation
and lightweight heads for worker selection and workflow actions, avoiding
autoregressive text generation for the fast routing path. Cloud workers such as
GLM, DeepSeek, Kimi, and Qwen Coder should be benchmarked as workers in the
pool, not treated as the orchestrator itself.

Terminal-Bench 2.1 should be the agentic harness after BigCodeBench, not a
replacement for BigCodeBench. Use BigCodeBench for cheap single-step
worker-selection labels; use Terminal-Bench for multi-turn worker switching,
tool-use, verification, and scaffold evaluation.

The Terminal-Bench track is now scoped as a metadata-only pilot first:
`research/evals/terminal_bench_2p1_pilot_plan.json`. Keep it held out until the
BigCodeBench-trained router has a stronger repeated dataset. The first pilot
should ask whether the active router can pick the initial worker for terminal
tasks and whether trajectory state features are needed before any adaptive
memory refresh consumes terminal traces.

The first Ollama Cloud three-task BigCodeBench smoke confirms that current
top-worker data is still too sparse for neural orchestrator training:
`ollama-cloud-qwen3-coder-480b` solved 1/3 and every other cloud worker solved
0/3. The next step remains data quality, not model complexity: expand positive
BigCodeBench outcomes and measure repeatability before training the logits-head
orchestrator.

Positive mining with `ollama-cloud-qwen3-coder-480b` over the 16 eligible tasks
found 4 solvable tasks. A faster cloud comparison on that subset produced
meaningful soft-target structure: Qwen solved 4/4, GLM and Kimi solved 3/4, and
DeepSeek V4 Pro solved 2/4. Keep using this mine-then-compare pattern until the
dataset contains enough diverse winners to justify logits-head training.

A two-sample repeatability smoke on the first two mined positives showed that
single-sample labels are brittle: Qwen and Kimi were stable at 4/4, DeepSeek V4
Pro was 3/4, and GLM was 2/4. Before logits-head training, add a repeated-sample
dataset converter so rewards can use empirical pass rates and latency
distributions instead of one-off pass/fail labels.

The repeated-sample dataset converter is now in place. Run it next on a larger
mined-positive subset so the first logits-head prototype can train against
empirical soft targets instead of one-off route outcomes.

The first local logits-head prototype now trains against the repeated-sample
soft-target dataset and writes a model artifact under `research/models/`. It is
a linear softmax head over prompt features, not a language-model backbone yet.
Treat this as proof of the trainable policy path; expand the repeated dataset
before attempting a sub-1B or LoRA orchestrator.

The logits-head path has now been expanded to all 4 mined Qwen-positive tasks,
with 32 repeated outcomes feeding a 4-task empirical routing dataset. The model
fits the soft targets cleanly, but every hard target is still Qwen Coder. The
next data goal is to find repeated benchmark slices where another worker is the
empirical winner under at least one reward objective.

Kimi mining over Qwen-negative tasks found two repeated non-Qwen winners:
`BigCodeBench/310` and `BigCodeBench/592`. The merged six-task dataset now has
mixed hard targets, 4 Qwen and 2 Kimi, and the logits router fits them. Continue
expanding with this pattern: mine failures from the current default worker,
repeat-compare candidate specialist wins, then retrain the logits head.

The logits router now reports leave-one-out evaluation. On the six-task
mixed-winner dataset it gets 5/6 held-out targets and misses `BigCodeBench/454`,
predicting Kimi where Qwen is the empirical target. Next data acquisition should
focus on Qwen-only and Kimi-favored filesystem tasks to make that boundary less
sparse.

The offset-99 filesystem scan added one GLM hard target (`BigCodeBench/963`) and
one broad-pass Qwen-latency target (`BigCodeBench/854`). The eight-task logits
router fits Qwen/Kimi/GLM targets, but LOO drops to 6/8 because GLM and Qwen-only
filesystem regions are still sparse. Continue data acquisition before moving to
a larger backbone.

Adaptive refresh now has a first gate: `tools/policy_refresh_gate.py` compares a
candidate logits-router report against a baseline using leave-one-out accuracy,
dataset size, and target-worker diversity. The eight-task model is promoted over
the six-task model with a warning because it adds GLM target diversity while
staying within the allowed LOO regression band. Add active/previous model
pointers and rollback bookkeeping next.

The active policy registry now records the promoted logits router, previous
policy pointer, and promotion history under `research/policies/active_policy.json`.
Rollback tooling exists. Next, runtime routing/evaluation should load the active
policy registry by default so learned-policy comparisons use the promoted model
instead of hardcoded artifact paths.

Runtime evaluation can now load the promoted model from the active policy
registry with `tools/evaluate_active_policy.py`. Next, include active-policy
metrics directly in the broader router baseline report so the promoted learned
policy is compared against strongest-worker, fastest-worker, family-router, and
nearest-neighbor baselines in one artifact.

Active-policy metrics are now included in the broader router comparison report
via `tools/train_router_baseline.py --active-policy-registry`. The next
evaluation gap is a separate held-out repeated dataset; current active-policy
and nearest-neighbor scores are still training-set comparisons.

The first held-out active-policy diagnostic ran two new repeated BigCodeBench
tasks outside the active eight-task training set. All four fast cloud workers
passed both tasks, so the target was latency-driven and Qwen won both rows. The
active logits router still solved both tasks, but routed both to Kimi, yielding
2/2 pass@1 and 0/2 target accuracy with higher latency. Next, collect more
latency-tie and specialist-win examples, then refresh the logits router before
moving to a larger backbone or Terminal-Bench.

Offset-99 specialist mining found no usable positives: GLM, DeepSeek, Kimi, and
Qwen all failed two samples on four Qwen/Kimi-negative tasks. Offset-125 then
added two stable broad-pass Qwen latency targets, but merging them into the
active dataset produced a ten-task logits-router candidate with poor
leave-one-out accuracy, so the refresh gate quarantined it. Keep the current
active policy; the next refresh needs either more non-Qwen specialist wins or a
latency-regret-aware objective/features rather than simply adding more Qwen
latency-tie rows.

Router reports now include `mean_target_latency_ms` and nonnegative
`mean_latency_regret_ms`. The held-out latency diagnostic shows the active
policy at 2/2 pass@1 but 2258.5 ms mean latency regret, while the ten-task
diagnostic shows 9/10 pass@1, 7/10 target accuracy, and 1138.4 ms mean latency
regret. Use this metric as a promotion guard for future refreshes.

Probe-gated latency calibration is now represented as a reusable policy
artifact and included in the normal baseline report. On the 37-task measured
slice it preserves active-router pass@1 while improving target accuracy from
0.7027 to 0.8919 and lowering mean latency regret from 1469.5 ms to 1127.2 ms.
The next validation step is a fresh held-out measured batch evaluated against
the saved probe-gated policy without retuning.

The saved probe-gated policy has now passed held-out replay validation on three
disjoint measured slices, preserving active-router pass@1 while improving
target accuracy and reducing latency regret on each. This makes it the current
conditional-verifier candidate. The next step should be a small live cloud
validation batch with the policy frozen.

The first small live cloud validation attempt was a negative control: all four
fresh tasks failed for all top-4 workers across two samples. The probe gate
correctly calibrated zero rows, but the run cannot validate the policy on solved
tasks. Future live validation must include a solvability screen before repeated
top-4 evaluation.

A follow-up solvability-screened live attempt ran one Qwen sample across twelve
fresh evaluator-friendly candidates and graduated zero tasks. This confirms the
screening process saves repeated top-4 spend, but generic fresh-task novelty is
still a poor source for live validation. Next live validation should draw from
known-positive or near-positive acquisition sources.

A known-positive live validation slice finally produced useful repeated data:
7 graduated tasks, 56 live outcome rows, and 37 passing samples. The frozen
probe-gated policy improved active-router target accuracy and latency regret
without changing active-router pass@1, but strongest/fastest single-worker
baselines solved all tasks. The conditional calibration mechanism is useful;
the active logits router is now the bottleneck. The next refresh should merge
this live dataset and gate against strongest-worker pass@1.

The adaptive refresh gate now has optional latency-regret thresholds. Rechecking
the quarantined ten-task candidate with corrected leave-one-out regret metrics
keeps it quarantined: target accuracy is 0.40, mean latency regret is 1556.1 ms,
and regret increased by 1034.0 ms over the active eight-task baseline. Future
promotion attempts should set both accuracy and latency-regret guards.

Adding explicit library one-hot features and a few low-level task keywords
recovered the ten-task refresh. The library-aware logits router is now the
active policy in `research/policies/active_policy.json`, trained on a ten-task
mixed-winner dataset with Qwen, Kimi, and GLM targets. It reaches 0.80 training
target accuracy, 0.90 training pass@1, and 518.6 ms training mean latency
regret; leave-one-out is weaker at 0.70 target accuracy, 0.80 pass@1, and
919.5 ms mean latency regret, but stayed within the current promotion gate.

The same active policy now fixes the two-task held-out broad-pass latency
diagnostic by routing both rows to Qwen, yielding 1.0 target accuracy, 1.0
pass@1, and 0.0 ms mean latency regret. This strengthens the case for richer
features and latency-regret-aware training before a larger backbone. The next
research loop should either collect more non-Qwen specialist wins or add a
latency-regret-aware training objective, then rerun held-out diagnostics before
starting the Terminal-Bench pilot.

The first latency-regret-aware training objective is now in place as
reward-tempered logits-router training. `tools/train_logits_router.py` supports
`--target-mode reward`, which trains against a softmax over stored worker
rewards rather than the already-materialized target distribution. On the active
ten-task dataset, reward temperature 0.10 improved leave-one-out target
accuracy from 0.70 to 0.80, pass@1 from 0.80 to 0.90, and mean latency regret
from 919.5 ms to 518.6 ms. The candidate passed the refresh gate with no
warnings and is now active. The next router work should automate this
temperature selection and add more non-Qwen specialist rows before moving to a
sub-1B or LoRA orchestrator.

Temperature selection is now automated by
`tools/select_logits_router_temperature.py`. It trains candidate temperatures,
runs each through the refresh gate, and selects the best promotable candidate by
leave-one-out target accuracy, pass@1, latency regret, and KL. On the active
ten-task dataset the selector chose temperature 0.10 again, matching the active
policy. This converts the previous hand-tuned objective into a repeatable
refresh step. Next, refreshes should use the selector after adding new measured
outcome rows.

Specialist mining found the first verified DeepSeek target:
`BigCodeBench/368`, a filesystem task involving `shutil`, `random`, and `os`.
DeepSeek solved it 2/2 while GLM, Kimi, and Qwen all failed 0/2. The new row
creates an 11-task diagnostic dataset with four target workers, but the router
refresh remains quarantined because leave-one-out latency regret rises to
1047.7 ms. This confirms the data direction but not the promotion. Mine more
DeepSeek-like rows before increasing model complexity or relaxing the gate.

The new similar-task selector shows that the current locally eligible
BigCodeBench pool has no fresh untested tasks similar to the DeepSeek
`BigCodeBench/368` row once existing routing records and outcome files are
excluded. The next data milestone should broaden eligibility rather than repeat
the exhausted pool: either add a dependency profile for more BigCodeBench tasks
or begin the tiny Terminal-Bench pilot for terminal/file-operation trajectories.

Dependency-gap analysis now gives a concrete path for broadening BigCodeBench:
out of 148 scanned rows, 117 are blocked by missing packages. The top four
packages, `pandas`, `numpy`, `matplotlib`, and `requests`, account for 89 unique
blocked rows. The next benchmark-environment step should create an isolated
profile for those packages, rerun canonical eligibility scans, and only then
spend cloud calls on the expanded pool.

The isolated top-four BigCodeBench profile has now been created and verified in
`.venv-bigcodebench`. With `pandas`, `numpy`, `matplotlib`, and `requests`
installed, the same 148-row scan yields 69 canonical-pass tasks instead of 29.
The next acquisition pass should sample from
`research/evals/bigcodebench_hard_top4_eligible_merged_tasks.json`, favoring
new plotting, data-science, filesystem, and network-adjacent tasks that are not
already in the routing dataset. Do not start the Terminal-Bench pilot until this
larger pool has produced repeated worker comparisons and a refreshed router
decision.

A fresh-batch selector now materializes diverse, unevaluated tasks from the
top-four pool while excluding existing routing records and raw outcomes. Its
first 8-task batch produced three Qwen-positive tasks, but the resulting
14-task router refresh stayed quarantined because leave-one-out accuracy and
latency regret regressed against the active ten-task policy. The next
acquisition pass should mine the remaining fresh top-four tasks with Kimi, GLM,
and DeepSeek first, rather than adding more Qwen-positive rows.

The first non-Qwen specialist mining pass found two positive tasks, but repeat
comparison showed both are broad-pass Qwen-latency rows. A 16-task diagnostic
refresh still stayed quarantined on latency regret even though target accuracy
reached the threshold. Continue using the non-Qwen pool as a mining filter, but
prioritize rows where Qwen is absent, unstable, or meaningfully slower before
another promotion attempt.

The fresh selector now supports a `hard` strategy for this phase. It found two
fresh Qwen-negative Kimi targets, `BigCodeBench/1004` and `BigCodeBench/1006`,
plus one noisy Qwen target, `BigCodeBench/760`. The resulting 19-task refresh
improves latency regret at low reward temperature but misses the promotion
accuracy gate by one small step. Next router work should improve features for
network/archive/request-style tasks before trying to promote the 19-task
dataset.

Network/archive/request interaction features are now in the router feature set,
and they recovered the 19-task refresh. The active policy now points to
`research/models/20260627-mixed-winner-19task-network-features-reward-t0p05-logits-router.json`
with a 19-task dataset and four target workers. This is still a lightweight
linear logits head, but it is now learning from real worker outcomes across
Qwen, Kimi, GLM, and DeepSeek targets. Next, run a held-out hard-batch
diagnostic before escalating to a neural or sub-1B orchestrator backbone.

The held-out hard-batch diagnostic found a concrete generalization failure. On
four excluded top-four BigCodeBench tasks with two samples per worker, only
`BigCodeBench/763` was solvable, and DeepSeek was the stable target. The active
router still selected Qwen for every row, yielding 0.0 pass@1 despite 0.75
target accuracy on the full held-out set. Before another policy promotion, add
a solvable-row gate to refresh selection so all-fail fastest-failure rows
cannot mask misses on rows that actually matter. The core evaluation reports
now expose solvable-row metrics, and refresh selection can enforce a minimum
leave-one-out solvable pass@1 threshold.

The first solvable-gated refresh has now promoted a 23-task router that includes
the hard-slice DeepSeek target. It fixes the `BigCodeBench/763` regression slice
and reaches 18/23 solved on the merged dataset, with 0.90 solvable pass@1 in
direct active-policy evaluation. This strengthens Milestone 3, but it does not
complete the small trainable orchestrator milestone: the active policy is still
a linear logits head over engineered features. Run one more fresh held-out
batch before escalating to a neural/sub-1B backbone or a verifier/abstain head.

That next fresh held-out batch found a new miss: `BigCodeBench/526` is a stable
GLM target while the active 23-task router still routes every row to Qwen. The
next step should not be another immediate single-row promotion. Add
solvability-aware evaluation/decision structure, such as a verifier/abstain
head or a second-stage capacity increase, then re-evaluate against the GLM
regression slice. The regression slices are now encoded in
`research/evals/router_regression_slices.json`.

An offline conditional-fallback evaluator now shows that a verifier-style second
attempt can pass both the DeepSeek and GLM regression slices with
`max_attempts=2`. This improves solvable coverage but increases latency and can
solve with a non-target worker. The next implementation should train or gate a
verifier/abstain head that decides when the second attempt is worth spending,
rather than always falling back after failure.

The first gated version uses router uncertainty: fallback only when the
top-vs-second worker margin is at most 0.10. It matches always-fallback solved
coverage on the active 23-task dataset while reducing mean latency from about
6.7s to 4.9s. Treat this as the prototype target for the future
fallback/verify logit head.

The cloud worker catalog is now refreshable from the live OpenAI-compatible
model list. The current generated pool adds candidate coverage for
`qwen3-coder-next`, `qwen3.5:397b`, and `gpt-oss:120b` while retaining the
already-measured Qwen Coder, Kimi, GLM, and DeepSeek workers. Treat these newer
candidates as acquisition targets, not active router evidence, until they have
repeatable BigCodeBench outcomes.

A bounded candidate acquisition run tested those three newer workers on the
DeepSeek `BigCodeBench/763` and GLM `BigCodeBench/526` regression slices. The
first attempt exposed an environment bug: running top-four dependency tasks
with base Python turns valid candidates into dependency failures. The valid
rerun used `.venv-bigcodebench`. `qwen3-coder-next` passed 1/2 samples on
`763` but failed 0/2 on `526`; `qwen3.5:397b` and `gpt-oss:120b` were 0/4.
Do not add these rows to the active router dataset yet. Keep the current
measured winners for these slices and use the new rows as calibration evidence
for candidate-screening and environment enforcement.

The real-worker runner now records evaluator environment provenance in summary
and JSONL outputs and can fail fast when required packages are missing. Future
top-four BigCodeBench runs should pass `--required-package numpy`,
`--required-package pandas`, and any other dependency-profile packages used by
the selected task batch. This prevents dependency errors from being mistaken for
model failures in training data.

The routing dataset converters now support `--required-evaluator-package` as a
second guard. This means both acquisition and conversion can reject rows from
the wrong evaluator environment before they influence soft targets or router
refreshes.

Outcome files now have a standalone conversion-readiness audit through
`tools/audit_outcome_rows.py`. Use it before building a routing dataset from any
new dependency-profile run. The audit intentionally marks older pre-provenance
outcome files as not ready when package requirements are supplied; those files
can stay as historical reports but should not be silently converted into new
training labels.

The catalog-candidate regression slice has now been rerun with evaluator
package provenance. The audit passes and a guarded routing dataset was built,
but it should remain calibration evidence rather than active training data:
all new candidates fail on the GLM `BigCodeBench/526` slice, and their
`BigCodeBench/763` passes are intermittent compared with the earlier stable
DeepSeek target. This reinforces the need for solvable-row and stability-aware
gates before merging candidate-only evidence into router refresh datasets.

A routing merge-readiness audit now enforces that decision mechanically. The
catalog-candidate provenance routing dataset validates structurally, but its
merge audit quarantines it because `BigCodeBench/526` is an all-fail fastest
failure row and the `BigCodeBench/763` target has only 0.5 pass rate. Future
merge workflows should run `tools/audit_routing_merge_readiness.py` before
`tools/merge_routing_datasets.py`.

`tools/merge_routing_datasets.py` now also supports `--require-merge-ready`.
Use that flag for router-refresh datasets so structurally valid but unstable
candidate evidence cannot be merged accidentally. Unguarded merge remains
available for historical inspection and explicitly negative datasets.

The M5 transition now has a repeatable readiness audit:
`research/programs/small_orchestrator_readiness.json`. The active 23-task
policy passes the worker-logits quality checks and the selected gated fallback
report is usable as a verifier/fallback teacher signal. The audit still blocks
small-orchestrator training because the repeated dataset is below the 50-task
threshold and explicit workflow-kind plus abstain-probability heads do not
exist yet. Continue M3 data acquisition and add the missing multi-head output
contract before sub-1B or LoRA orchestrator training.

The multi-head output contract now exists at
`research/models/20260627-active-multi-head-orchestrator-contract.json`. It
defines the M5 action surface with worker-distribution, workflow-kind,
verifier-probability, and abstain-probability heads. The readiness gate now
validates that contract, so the workflow/abstain blocker is cleared. The
remaining M5 blocker is data volume: the active repeated routing dataset has
23 tasks and must reach at least 50 stable rows before backbone training.

The next data-volume wave is now planned in
`research/programs/acquisition_to_50_plan.json`, with selected tasks written to
`research/evals/bigcodebench_hard_acquisition_to_50_wave1_tasks.json`. It
overselects 41 fresh top-four BigCodeBench tasks to cover the 27-row gap after
merge-readiness filtering. Run the manifest command sequence with the
`.venv-bigcodebench` evaluator, require `numpy` and `pandas`, then only merge
rows that pass the outcome audit, routing conversion, and merge-readiness gate.

Wave 1 has run across the measured top-four pool. The outcome audit passed with
328 rows, and the repeated converter produced 41 routing records. The raw
dataset was not merge-ready: 29 rows were all-fail and 3 target rows were
unstable. Filtering yielded 9 merge-ready rows and a 32-task candidate dataset,
but the candidate router was quarantined by the refresh gate because LOO target
accuracy dropped to 0.625 and mean latency regret rose to about 1000 ms. Keep
the active 23-task policy unchanged. Next acquisition should either collect
more stable specialist rows before retraining or add feature/capacity changes
that specifically help the new DeepSeek/GLM/Qwen latency rows.

Wave 2 exhausted the current fresh canonical top-four pool under the existing
dependency profile. After excluding every active, wave1, and prior-outcome task,
only two fresh specialist candidates remained. The repeated comparison produced
one merge-ready row: `BigCodeBench/162`, a broad-pass Qwen latency target.
`BigCodeBench/208` was all-fail and filtered out. Merging active, wave1, and
wave2 rows created a 33-task candidate dataset, but the router was quarantined:
LOO target accuracy dropped to 0.636 and mean latency regret rose to about 970
ms. Keep the active 23-task policy unchanged. The next path to 50 stable rows
should widen the acquisition source, either through a broader benchmark
dependency profile or a small held-out agentic/secondary benchmark slice.

The broader BigCodeBench dependency profile is now validated as the next data
source. Adding `sklearn`, `scipy`, `seaborn`, and `bs4` to the isolated evaluator
profile made 19 of the first 30 hard-subset rows canonical-pass, and after
excluding all prior evidence produced 5 merge-ready fresh rows. Those rows added
DeepSeek, Kimi, GLM, and Qwen targets and moved the stable candidate set to 38
tasks. The 38-task router was quarantined because LOO target accuracy dropped to
0.474 and mean latency regret rose to about 4430 ms. Keep the active 23-task
policy unchanged. Continue expanded-profile acquisition toward the 50-row M5
threshold, but add better dependency/domain and latency-regret features before
expecting a larger candidate router to promote.

The repeated-routing converter now uses evaluator package provenance when
deriving prompt `missing_libraries`, so rows evaluated in `.venv-bigcodebench`
are not mislabeled by the converter process environment. A simple
category-library interaction feature attempt was tested on the 38-task expanded
candidate and discarded: it improved in-sample fit but worsened leave-one-out
target accuracy and latency regret. The final provenance-fixed 38-task router
also remains quarantined. Keep the active 23-task policy unchanged and continue
data acquisition before increasing feature complexity.

Expanded-profile wave 2 scanned offset 30 and found 18 canonical-pass tasks, but
after excluding previous evidence only 7 fresh tasks remained. Repeated
top-four comparison yielded 2 merge-ready rows (`BigCodeBench/302` and
`BigCodeBench/445`) and 5 all-fail rows. Both kept rows are Qwen latency
targets, moving the stable candidate set from 38 to 40 rows. The 40-task router
remains quarantined with LOO target accuracy 0.500 and mean latency regret about
4208 ms. Continue acquisition toward 50 rows, but bias selection toward
non-Qwen specialist opportunities or additional dependency slices rather than
more broad-pass Qwen latency rows.

Expanded-profile wave 3 scanned offset 60 and found 18 canonical-pass tasks.
After excluding previous evidence, 6 fresh tasks were measured across the
top-four worker pool with two samples each. Five rows were merge-ready and all
five selected Qwen as the stable reward target, moving the stable candidate set
from 40 to 45 rows. The resulting 45-task router remains quarantined: LOO target
accuracy was 0.644 and mean latency regret was about 1746 ms, despite solvable
pass-at-1 reaching 0.857. Keep the active 23-task policy unchanged. The next
acquisition wave should finish the 50-row data-volume gate while actively
looking for non-Qwen specialist labels or clearer latency tradeoffs.

Expanded-profile wave 4 completed the 50-row stable repeated-routing threshold.
Across offset 90, 115, and 145 acquisition slices, it added 5 merge-ready rows:
DeepSeek targets for `BigCodeBench/752` and `BigCodeBench/1053`, a Kimi target
for `BigCodeBench/1013`, and Qwen targets for `BigCodeBench/969` and
`BigCodeBench/1124`. The resulting
`research/datasets/20260627-expanded-profile-wave4-50task-routing.jsonl` is the
first dataset that satisfies the M5 data-volume gate. The trained 50-task logits
router remains quarantined, however: LOO target accuracy was 0.620, solvable
pass-at-1 was 0.872, and mean latency regret was about 3610 ms. Keep the active
23-task policy unchanged. The next step should use the 50-task dataset as an
offline small-orchestrator training substrate while treating specialist target
accuracy and latency-regret reduction as the gating problem.

The first M5 supervised substrate now exists at
`research/datasets/20260628-m5-small-orchestrator-substrate-50task.jsonl`, with
its manifest at
`research/datasets/20260628-m5-small-orchestrator-substrate-50task-manifest.json`.
It converts the 50 repeated routing records into multi-head examples containing
worker-distribution, workflow-kind, verifier-probability, and
abstain-probability targets plus chat-style messages for later fine-tuning.
The manifest has 50 records: 47 direct workflow labels and 3
verify-then-fallback/abstain rows. This is an offline training substrate, not a
promoted policy. Next, train a local candidate against this substrate and reuse
the policy gate before any active-policy refresh.

The first local trainable multi-head orchestrator artifact now exists at
`research/models/20260628-m5-offline-multihead-50task.json`, trained by
`tools/train_multi_head_orchestrator.py` from the 50-row substrate. It emits the
same worker, workflow, verifier, and abstain heads required by the M5 contract.
The candidate is quarantined by
`research/evals/results/20260628-m5-offline-multihead-50task-policy-gate.json`:
LOO target accuracy is 0.620, solvable pass-at-1 is 0.872, and mean latency
regret is about 3610 ms. This validates the offline small-orchestrator training
path but does not justify active-policy promotion. Next modeling work should
target specialist-sensitive features or a richer local backbone, then reuse the
same policy gate.

The first adaptive-memory refresh cycle now exists at
`research/refreshes/20260628-m5-offline-multihead-refresh-cycle.json`, with a
privacy manifest at
`research/refreshes/20260628-m5-offline-multihead-privacy.json`. It packages the
50-row distilled substrate, the offline multi-head candidate, the policy gate,
the active-policy rollback point, and privacy/memory-scope guardrails into one
auditable refresh decision. All guardrails pass, but the cycle decision is
`quarantine` because the candidate failed the policy gate. This completes the
first evaluated refresh-cycle path without changing the active policy.

The live known-positive validation slice has now been merged into a 30-task
live-augmented dataset. A frozen probe-gated policy preserves active pass@1
while improving target accuracy and latency regret on that merged report, but a
fresh raw logits-router retrain remains quarantined. The new refresh gate also
checks candidate LOO pass@1 against the strongest single-worker pass@1; the
best fresh candidates pass that solvability check, but still fail target
accuracy and latency-regret preservation. Keep the strongest-worker pass gate
and the 30-task artifacts, but do not promote the raw refresh. The next modeling
step should train a pass-first or conditional/probe-gated policy against the
frozen probe-gated result.

Refresh selection can now use a named operational-policy reference from a
baseline report. On the 30-task live-augmented sweep, the
`probe-gated-latency-calibrated-logits-router` became the explicit
`preserve_accuracy` reference, raising the promotion bar to 0.8333 target
accuracy, 0.8519 solvable pass@1, and 251.4 ms latency regret. All raw
temperature-sweep retrains remain quarantined under that stronger comparison.
Use this operational-reference gate for the next conditional/probe-gated
candidate before any active-policy update.

A first operational workflow-policy gate now exists for candidates without
leave-one-out router metrics. The 30-task margin-gated fallback candidate
passes regression slices and improves pass@1 to 0.8333 plus solvable pass@1 to
0.9259, but it is quarantined against the probe-gated reference because mean
latency regret rises to 1815.3 ms. This confirms the next conditional policy
should be value-aware: it must predict when a second attempt is worth the extra
latency, not only when router confidence is low.

The first learned second-attempt value head is now non-leaky and selectable as a
policy artifact. On the same 30-task live-augmented dataset it reaches pass@1
0.8333 and solvable pass@1 0.9259, but remains quarantined against the
probe-gated reference because target accuracy drops to 0.7667 and latency
regret rises to 2087.4 ms. Keep the value-head tooling, but mine more fallback
opportunities before expecting learned value gating to promote.

A historical fallback-opportunity corpus now exists across all current measured
routing datasets. It contains 322 top-fail fallback opportunities, but only 86
unique tasks after dedupe and 16 positive deduped tasks. The mined fallback head
fits the deduped training set moderately (F1 0.5806) but leave-one-out F1 drops
to 0.2222, so the corpus is better used for acquisition planning than immediate
policy promotion. Next acquisition should target novel fallback-positive tasks,
especially cheap specialist rescues, rather than adding more duplicate merged
rows.

A corpus-guided acquisition batch now exists at
`research/evals/bigcodebench_hard_corpus_guided_fallback_batch8_tasks.json`.
The selector scored 65 fresh eligible candidates using fallback-positive
similarity, useful-second similarity, active-router margin, preferred alternate
workers, environment risk, and hard-negative similarity. It selected eight
mostly filesystem/text/general tasks around Kimi/Qwen decision boundaries. Next,
evaluate this batch with repeated top-worker calls and only use it for a value
head refresh if it adds genuine fallback-positive or cheap second-attempt
evidence.

The corpus-guided batch has now been evaluated with repeated top-four worker
calls. It added eight routing records, including six solvable rows and target
diversity across Qwen, GLM, and Kimi. It did not add useful active-router
fallback positives: the active router's top failures on the slice were the two
universal failures. A 38-task raw logits-router refresh that merged this slice
with the live-augmented dataset remains quarantined against the probe-gated
operational reference. Next work should target active-router first-attempt
failures that are solvable by a ranked alternate, while preserving the
latency-calibration lesson from this batch.

An active-rescue acquisition selector now targets that missing signal directly.
It conditions on the active router's predicted top and second workers, rewards
historical same-pair rescue evidence, and penalizes same-pair hard negatives.
The next selected eight-task batch is
`research/evals/20260628-active-rescue-acquisition-batch8-tasks.json`. Evaluate
it with repeated top-four calls and judge the run by mined useful fallback
positives, not by immediate active-policy promotion.

The active-rescue batch was evaluated with repeated top-four calls and produced
another useful negative for fallback acquisition: four tasks were solvable, four
were universal failures, and active-router top-fail mining produced only hard
negatives. This confirms that more task-level selector work has diminishing
returns for now.

The current task-level orchestrator has now been trained on a 66-record
substrate at
`research/datasets/20260628-m5-current-task-66task-substrate.jsonl`.
The trained checkpoint is
`research/models/20260628-m5-current-task-66task-multihead.json`.
It is publishable as a research checkpoint, but not promotable: leave-one-out
target accuracy is 0.5606 and mean latency regret is 2463.5 ms. Finish public
repo setup with this checkpoint, then continue improving task-level specialist
coverage before starting turn-level agentic training.

Turn-level substrate scaffolding is now allowed as preparation, but not as the
active training track. The converter should accept only sanitized, summary-level
terminal trajectories and emit per-turn examples with worker, workflow, action,
verifier, repair, stop, and memory-update heads. Do not train or promote a
turn-level policy until task-level reliability improves and real multi-turn
trajectories are available.

The prompt-set evaluator now separates missing benchmark runtime dependencies
from ordinary model test failures. A current-environment preflight manifest at
`research/evals/20260628-current-env-evaluable-tasks.json` filters the 40-task
comparison-source union down to 27 locally evaluable tasks. Use this manifest
for the next live task-level acquisition run unless we first add a pinned
benchmark environment with packages such as `pandas`, `matplotlib`, `numpy`,
`requests`, `bs4`, and `sklearn`.

Clarified Milestone 5 architecture: the linear logits router is only the
baseline and data-loop validator. The intended small-orchestrator track should
use a Qwen-small style local language-model backbone with explicit heads over a
decision hidden state. Train worker, workflow, verifier, and abstain heads
against measured soft routing targets first, with the backbone frozen; only try
LoRA/adapters after the heads beat the linear router on held-out task-level
routing.

The first live comparison drawn from the current-environment evaluable task
manifest produced six fully evaluable rows. The trained router solved 1/3 while
fixed Qwen solved 0/3, again through GLM on `BigCodeBench-339`; however mean
latency was much worse for the router. Keep this as clean specialist evidence,
not a promotion signal. Since Qwen-small head training is blocked locally by
missing ML dependencies, the next best progress is either installing
`torch`/`transformers` or continuing clean task-level acquisition.

The Qwen training readiness audit now records the exact local blocker:
`research/models/20260628-qwen-training-readiness.json`. The active Python is
3.14.4 and the ML stack is absent, so the first real head-training run should
use Python 3.11/3.12 with `.[qwen-train]`, or move to a GPU/MLX machine.

That local path now works: `.venv-qwen-train` uses Python 3.11.14 with PyTorch,
Transformers, and Apple MPS available. The first Qwen-small frozen-head smoke
trained one epoch over 66 rows and wrote
`research/models/20260628-qwen-small-logits-orchestrator-smoke/qwen_logits_heads.pt`.
Reloading that checkpoint and evaluating it on the same rows gives worker
accuracy 0.3182 and workflow accuracy 0.8636. Next, publish the prepared Hugging
Face dataset/model artifacts once auth is available, then run longer training
and held-out evaluation before any promotion discussion.
