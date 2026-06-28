# Benchmark Strategy

The first benchmark should be small enough to run repeatedly and rigorous enough
to expose whether orchestration adds value over a single strong worker.

## Recommendation

Start with BigCodeBench-Hard in instruct mode.

Use BigCodeBench as the measured routing-data source, not as a leaderboard
claim. The orchestrator needs per-task worker outcomes in the local harness; see
`docs/measured_routing_data.md`.

Why:

- It is code-focused, so evaluation can use executable tests instead of only
  subjective judging.
- The hard subset is about 150 tasks, which is manageable for iteration.
- The tasks are practical code-generation problems with richer library and API
  use than classic toy benchmarks.
- It supports multiple backends and stores generated samples and evaluation
  results as files that can be compared across runs.
- It can be downsampled into a smoke set before spending money on a full run.

## Why Not Start Elsewhere

- SWE-style repository benchmarks are closer to real software work, but they are
  expensive, slow, and noisy for the first orchestration loop.
- Competitive-programming benchmarks are useful for algorithmic reasoning, but
  they are less representative of broad tool-oriented coding work.
- General reasoning benchmarks are cheap to prompt but harder to verify without
  judge bias.
- Classic small coding benchmarks are fast, but many are saturated and less
  useful for comparing strong workers.

## Evaluation Ladder

1. Smoke set: 10 handpicked tasks from the hard subset.
2. Pilot set: 30-50 tasks stratified by task type and difficulty.
3. Full hard subset: about 150 tasks.
4. Agentic harness pilot: Terminal-Bench 2.1 subset once single-step routing has
   enough positive labels.
5. Second non-code benchmark only after the coordinator beats baselines on the
   first two code/agentic harnesses.

## Baselines

Every orchestration result must be compared against:

- best single top-tier worker
- cheapest acceptable worker
- strongest open-weight or local worker available
- simple rule router
- learned coordinator
- learned coordinator with conditional verifier

## Metrics

Primary:

- pass@1
- cost per solved task
- latency per solved task
- abstention rate

Secondary:

- verifier intervention rate
- routing distribution by task family
- failure mode categories
- improvement over best single worker
- degradation cases where orchestration hurts

## Run Policy

Do not start with a full benchmark run. First build the harness against a tiny
sample and verify that:

- prompts are normalized
- worker outputs are captured
- generated code is evaluated reproducibly
- cost and latency metadata are recorded
- failed evaluations are resumable
- outputs are comparable across workers and workflows
- evaluator Python and required package availability are recorded with outcome
  rows

## Dataset Expansion Strategy

The next orchestrator training run should be data-led. Treat new model
architecture as useful only when the training/evaluation data has enough worker
diversity to measure it.

Required fields for any imported or self-generated routing row:

- task text or normalized prompt
- candidate worker/model identity
- provider binding, if different from canonical worker identity
- executable score, pass/fail, human preference, or other comparable reward
- latency and approximate cost when available
- failure mode or evaluator error when available
- source/license metadata
- split key that prevents near-duplicate leakage into held-out evaluation

Acquisition order:

1. Search for existing public per-sample model outcome datasets. Prefer sources
   with executable pass/fail labels or pairwise preferences and compatible
   licenses. Reject aggregate-only leaderboards for training labels.
2. Convert compatible public data into a neutral worker-outcome schema using
   canonical worker IDs and source metadata.
3. Fill gaps with our own screened benchmark runs. Use cheap solvability screens
   first, then repeated top-k comparisons only on likely-positive tasks.
4. Keep single-step code routing data separate from terminal/agent trajectories
   until turn-level routing has its own schema and evaluation gate.
5. Promote a new training dataset only when it adds either more tasks, more
   empirical winner classes, or better repeated-sample confidence without
   weakening held-out split hygiene.

Candidate source families to audit:

- code benchmark execution traces with per-sample pass/fail outcomes
- pairwise model-comparison datasets with prompt and winner labels
- public agent benchmark trajectories with final reward and tool traces
- local Codex/Claude traces only after sanitization, consent, and a separate
  privacy review

If public data is insufficient, continue creating our own data with the current
mine-screen-repeat pattern. The cost-control rule is simple: do not run all
workers on all tasks until a cheap screen says the task can produce a useful
positive or disagreement signal.

## Dynamic Worker Pool Strategy

Worker labels must become registry-backed rather than provider-string-backed.
The orchestrator should learn canonical worker IDs, while runtime configuration
maps those IDs to Ollama, local, or OpenAI-compatible provider endpoints.

Inference should support a worker mask:

- unavailable workers
- muted workers
- user-disallowed workers
- workers outside cost/latency policy
- deprecated workers kept only for checkpoint compatibility

Masked workers should receive impossible logits before softmax, so an existing
checkpoint can route over the currently available subset. New workers should be
introduced through a versioned expanded-head refresh: copy weights for unchanged
labels, initialize new labels from priors or metadata, collect measured outcomes,
train heads, and promote only if held-out routing improves or the new worker
wins a meaningful measured slice.

## Current External Smoke Status

The BigCodeBench-Hard smoke materializer writes a 10-task instruct-mode file at
`research/evals/bigcodebench_hard_smoke_tasks.json`. The first one-task local
Ollama probe produced zero passes across the current worker pool, while the
canonical solution passed under the same adapter. Treat this as a harness
success and a task-selection warning: run a 3-task mini-pilot with varied prompt
and library characteristics before spending time on the full 10-task smoke set.

The mini-pilot selector now writes:

- `research/evals/bigcodebench_hard_minipilot_tasks.json`
- `research/evals/bigcodebench_hard_minipilot_report.json`

The current selected tasks are `BigCodeBench/15`, `BigCodeBench/19`, and
`BigCodeBench/13`. They are the only canonical-pass tasks in the first 10-task
slice under the current local Python environment. The data-science and Flask
tasks are excluded for now because their official solutions fail locally due to
missing packages, which would confuse environment gaps with model failures.

The 3-task mini-pilot has now run against the current Ollama worker pool:

- qwen3-1.7b: 1/3, mean latency 81506 ms
- qwen3-4b-instruct: 1/3, mean latency 10833 ms
- lfm2.5-1.2b: 0/3, mean latency 3277 ms

Only `BigCodeBench/19` was solved, and qwen3-4b-instruct matched qwen3-1.7b on
that task with much lower latency. The next benchmark step should expand the
canonical-pass task pool before moving to a neural orchestrator.

The eligible-pool scanner now writes:

- `research/evals/bigcodebench_hard_eligible_tasks.json`
- `research/evals/bigcodebench_hard_eligible_report.json`

It found 8 canonical-pass tasks after probing 44 rows. This gives the next
external run a broader standard-library/system-task slice without installing
data-science or web dependencies.

The 8-task eligible pool has now run across the current Ollama worker pool:

- qwen3-1.7b: 1/8, mean latency 69781 ms
- qwen3-4b-instruct: 1/8, mean latency 8621 ms
- lfm2.5-1.2b: 0/8, mean latency 2411 ms

Only `BigCodeBench/19` passed. qwen3-4b-instruct is the strongest single-worker
baseline on this slice because it ties qwen3-1.7b on pass count with much lower
latency. The external dataset is still too failure-heavy for neural router
training.

The first Ollama Cloud three-task smoke run has now run across GLM, DeepSeek,
Kimi, and Qwen Coder cloud workers:

- ollama-cloud-qwen3-coder-480b: 1/3, mean latency 2799.7 ms
- ollama-cloud-glm-5.2: 0/3, mean latency 6598.0 ms
- ollama-cloud-deepseek-v4-pro: 0/3, mean latency 11699.0 ms
- ollama-cloud-deepseek-v3.2: 0/3, mean latency 77767.0 ms
- ollama-cloud-kimi-k2.7-code: 0/3, mean latency 11321.3 ms

This gives a real top-worker comparison but still collapses to one positive
worker, so it is not yet sufficient for neural router training. Continue
expanding positive BigCodeBench outcomes before the logits-head orchestrator.

The next positive-mining run evaluated `ollama-cloud-qwen3-coder-480b` across
all 16 canonical-pass eligible tasks and found 4 positives: `BigCodeBench/13`,
`BigCodeBench/19`, `BigCodeBench/454`, and `BigCodeBench/777`. A faster
multi-worker comparison on those mined positives produced:

- ollama-cloud-qwen3-coder-480b: 4/4, mean latency 2662.8 ms
- ollama-cloud-glm-5.2: 3/4, mean latency 10217.2 ms
- ollama-cloud-kimi-k2.7-code: 3/4, mean latency 13753.8 ms
- ollama-cloud-deepseek-v4-pro: 2/4, mean latency 13719.8 ms

Use this two-stage pattern for the next pilot: mine positives with the strongest
fast worker, then compare a broader worker pool only on the mined subset. This
reduces wasted calls while preserving the routing signal needed for soft target
distributions.

A repeatability smoke on the first two mined positives ran two samples per
worker/task across the faster cloud pool:

- ollama-cloud-qwen3-coder-480b: 4/4, mean latency 2831.8 ms
- ollama-cloud-kimi-k2.7-code: 4/4, mean latency 12347.0 ms
- ollama-cloud-deepseek-v4-pro: 3/4, mean latency 19483.2 ms
- ollama-cloud-glm-5.2: 2/4, mean latency 13897.2 ms

This shows that single-sample labels are too brittle for training. Future
training datasets should aggregate repeated samples into empirical pass rates
and latency-adjusted rewards before fitting the logits-head orchestrator.

The first logits-head prototype has been trained on the two-task repeated
dataset. It is a local linear softmax head over prompt features, emits one logit
per worker, and matches the empirical soft-target argmax on the tiny training
set. This proves the trainable-policy path, but it should not be treated as a
benchmark result until the repeated dataset is much larger and has diverse
winning workers.

The repeated dataset has now expanded to all 4 mined Qwen-positive tasks, with 2
samples per worker/task. Qwen remains the hard target for every task, while Kimi
and the other workers provide useful soft-target mass only on some tasks. The
next benchmark selection criterion should be diversity of empirical winners, not
just more Qwen-positive examples.

Mining the 12 Qwen-negative eligible tasks with Kimi found 2 non-Qwen positives:
`BigCodeBench/310` and `BigCodeBench/592`. A repeated comparison confirmed Kimi
as the empirical winner on both. The resulting six-task dataset has mixed hard
targets, 4 for Qwen and 2 for Kimi, and the logits router fits those targets.
Use this as the default data-expansion pattern: mine failures from the current
default worker with a specialist, then repeat-compare candidates before adding
them to the training set.

The logits router now includes leave-one-out evaluation. On the six-task
mixed-winner dataset it reaches 5/6 held-out target accuracy and 5/6 pass@1. The
miss is `BigCodeBench/454`, a Qwen-only target that the held-out model routes to
Kimi. Treat this as the next benchmark-selection hint: add more filesystem tasks
that separate Qwen-only wins from Kimi wins.

The offset-99 eligible scan added 8 new mostly filesystem/subprocess tasks.
Repeated comparison of the apparent Qwen-only candidates showed that
`BigCodeBench/854` is broad-pass and Qwen wins by latency, while
`BigCodeBench/963` is the first GLM hard target. The eight-task logits router
fits Qwen/Kimi/GLM training targets, but leave-one-out drops to 6/8 and misses
454 and 963. Keep mining filesystem tasks before increasing model complexity.

Policy refreshes are now gated. A candidate router must include leave-one-out
metrics and is compared against a baseline for minimum LOO accuracy, bounded LOO
regression, task count, and target-worker diversity. The eight-task model is
promoted over the six-task model with a warning because it adds GLM diversity
while staying within the allowed LOO accuracy drop.

Promoted policies are now written to `research/policies/active_policy.json`,
including active model, active dataset, previous policy, and promotion history.
The next benchmark reports should evaluate the active learned policy by loading
that registry, so model promotion changes the policy under test without editing
commands by hand.

The active learned policy can now be evaluated from the registry with
`tools/evaluate_active_policy.py`. Add this active-policy result into future
router reports so learned-policy performance is compared in the same artifact as
strongest-worker, fastest-worker, family-router, and nearest-neighbor baselines.

The baseline report now accepts `--active-policy-registry` and includes the
promoted learned policy. On the active eight-task dataset the active logits
router matches oracle, while simple family/fastest baselines solve 5/8 and the
strongest single worker solves 6/8. Treat this as an operational comparison, not
a final benchmark claim, because the learned policy is evaluated on its training
dataset.

The baseline report also accepts a saved probe-gated latency-calibration policy.
On the 37-task measured routing slice, the probe-gated router preserved the
active logits router's pass@1 while improving target accuracy and latency
regret. Treat this as a deployable-shaped diagnostic, not a leaderboard claim,
until it is run on a fresh held-out measured batch with frozen probe and
calibration settings.

The frozen probe-gated policy also improved target accuracy and latency regret
on three disjoint held-out replay slices without changing pass@1. This supports
using it as the current conditional-verifier candidate, while still requiring a
small live cloud validation batch before treating it as an external benchmark
result.

The first fresh live validation batch produced only universal failures, so it is
a selection negative control rather than a policy validation. Before spending a
full repeated top-4 run, screen fresh tasks for solvability with a cheap worker
probe or canonical-environment pass.

The first solvability-screened candidate batch also graduated zero tasks from a
Qwen one-sample screen. Continue using the screen gate, but source validation
tasks from known-positive neighborhoods rather than generic fresh novelty.

A known-positive live validation slice produced usable repeated top-4 data. The
probe-gated policy improved over the active logits router on target accuracy and
latency regret, but did not beat strongest/fastest single-worker pass@1. Future
promotion gates should include strongest-worker pass@1 as a hard comparison,
not just active-router deltas.

A held-out repeated diagnostic on `BigCodeBench/906` and `BigCodeBench/928`
showed a different failure mode. Every fast cloud worker passed every sample,
but Qwen was the empirical target on both rows because it was much faster. The
active logits router routed both tasks to Kimi: pass@1 stayed 2/2, target
accuracy was 0/2, and mean latency rose to 9050.5 ms versus Qwen's 2372.25 ms
overall mean on the same run. This means the next BigCodeBench work should add
latency-tie examples and specialist-win examples before escalating to a larger
orchestrator backbone.

The next mining attempt split into two lessons. Offset-99 Qwen/Kimi-negative
tasks were also negative for GLM and DeepSeek across two samples, so they should
remain hard-negative evidence rather than training rows. Offset-125 provided two
stable broad-pass Qwen latency targets, but a ten-task logits-router refresh was
quarantined because leave-one-out target accuracy dropped from 0.75 to 0.40.
Future BigCodeBench data acquisition should prioritize non-Qwen specialist wins
or add latency-regret-aware training before another promotion attempt.

Router reports now include latency regret against the empirical target worker.
This prevents a policy from looking acceptable on pass@1 while silently choosing
a much slower worker on broad-pass tasks. The current active policy has 2258.5
ms mean latency regret on the two-task held-out latency diagnostic and 1138.4 ms
on the ten-task merged diagnostic set.

Policy refresh gating now supports latency-regret thresholds. The ten-task
candidate refresh remains quarantined when gated with a 1000 ms maximum LOO
latency regret and 500 ms maximum regret increase: it reaches only 0.40 LOO
target accuracy and 1556.1 ms mean LOO latency regret.

The ten-task refresh was recovered by adding explicit library features and more
low-level task keywords to the prompt feature extractor. The promoted
library-aware logits router now uses the ten-task mixed-winner dataset with
Qwen, Kimi, and GLM targets. On the ten-task training comparison it reaches
0.80 target accuracy, 0.90 pass@1, and 518.6 ms mean latency regret. Its
leave-one-out gate reaches 0.70 target accuracy, 0.80 pass@1, and 919.5 ms
mean latency regret, inside the current promotion thresholds but still noisy.

The held-out broad-pass latency diagnostic on `BigCodeBench/906` and
`BigCodeBench/928` now routes both rows to Qwen, giving 1.0 target accuracy,
1.0 pass@1, and 0.0 ms mean latency regret. Treat this as a repaired failure
mode, not as a general benchmark win. The next BigCodeBench data should add
more non-Qwen specialist wins and more held-out broad-pass latency rows before
escalating to a larger orchestrator backbone.

The active router now trains with a reward-tempered objective. Training against
a softmax over stored worker rewards at temperature 0.10 improved leave-one-out
metrics on the same ten-task dataset from 0.70 target accuracy, 0.80 pass@1,
and 919.5 ms latency regret to 0.80 target accuracy, 0.90 pass@1, and 518.6 ms
latency regret. Higher temperatures of 0.20 and 0.50 did not improve the prior
LOO behavior. The reward-tempered router is now active through the policy
registry, and the held-out broad-pass latency diagnostic remains at 0.0 ms
latency regret.

Reward-temperature selection is now automated. The selector trains each
candidate temperature, applies the same refresh gate, and picks the best
promotable candidate by leave-one-out target accuracy, pass@1, latency regret,
and KL. On the active ten-task dataset it selected 0.10 again. Temperatures 0.20
and 0.50 are now allowed through the gate with warnings, but ranked below 0.10
because they regress accuracy and latency regret.

A bounded specialist-mining pass over ten Qwen-negative and Kimi-negative tasks
found one verified DeepSeek target: `BigCodeBench/368`. DeepSeek solved it 2/2
in repeat comparison while GLM, Kimi, and Qwen all failed 0/2. The row was added
to an 11-task diagnostic dataset, but the resulting router refresh was
quarantined. The best reward-temperature candidates reached 0.7273
leave-one-out target accuracy, 0.8182 pass@1, and 1047.7 ms mean latency regret,
exceeding the current latency-regret gate. Keep the DeepSeek row as evidence,
but mine more DeepSeek-like filesystem rows before promotion.

A similar-task selector now ranks candidate tasks by overlap with the
`BigCodeBench/368` library/category signature and can exclude rows already in
routing datasets or outcome files. On the current known eligible pool, every
fresh DeepSeek-like candidate is exhausted: after excluding the active 11-task
dataset and prior outcomes, the selector found zero untested candidates. The
next data step should expand the eligible source pool rather than rerun the
same BigCodeBench rows.

The existing canonical probe reports show that eligibility is mainly blocked by
missing packages. Across 148 scanned BigCodeBench-Hard rows, 29 are locally
eligible, 117 are dependency-blocked, and only 2 are non-dependency failures.
The largest blockers are `pandas` (42 rows), `numpy` (23), `matplotlib` (15),
and `requests` (9). A benchmark-specific dependency profile with those four
packages could potentially unlock 89 unique rows. Keep this out of the core
project dependencies; it belongs in an isolated benchmark environment.

The isolated top-four benchmark profile has now been tested in
`.venv-bigcodebench` with `pandas`, `numpy`, `matplotlib`, and `requests`.
Rerunning the same 148-row BigCodeBench-Hard scan increased canonical-pass
eligibility from 29 rows to 69 rows. The remaining dependency-blocked set fell
from 117 unique rows to 72, with the next blockers now led by `scikit-learn`
(12 rows), `scipy` (11), `seaborn` (8), and `bs4` (6). Use the merged top-four
eligible task file for the next data-acquisition pass before adding another
dependency profile.

## Agentic Harness Target

Terminal-Bench 2.1 should be the next harness family after BigCodeBench produces
enough single-step routing labels. It evaluates agents in a sandboxed terminal
environment across realistic software engineering, system administration, data
processing, model training, and security-flavored tasks.

Use it for the second-stage orchestrator question:

- which worker should drive an interactive terminal task?
- when should the orchestrator switch workers?
- when should it verify, repair, or stop?
- how much does the harness/scaffold change worker performance?

Do not replace the current BigCodeBench path with Terminal-Bench. BigCodeBench
is still the cheaper source of clean worker-selection labels. Terminal-Bench
belongs to the end-to-end agentic trajectory phase, where the orchestrator
learns from tool calls, file edits, tests, and terminal feedback.

The first Terminal-Bench step should be a tiny reproducibility pilot, not a
leaderboard run. Select a handful of Terminal-Bench 2.1 tasks only after the
BigCodeBench top-four pool has produced a larger repeated routing dataset. The
pilot should compare single-worker agent runs against an orchestrator-selected
worker/workflow policy and record whether state/history features are needed.
The concrete pilot plan now lives at
`research/evals/terminal_bench_2p1_pilot_plan.json`. Selection should use only
task metadata such as id, category, difficulty, and tags; do not copy task
instructions, oracle solutions, or verifier code into mempool training corpora.
Use `tools/extract_terminal_bench_metadata.py` to sanitize a local export or
task-directory checkout before selecting pilot tasks.
Trajectory records must validate against
`research/evals/terminal_bench_trajectory_schema.md` before they are used in
reports or future training conversions.
The initial metric is not leaderboard rank. It is whether the active
BigCodeBench-trained router is a useful initial-worker selector for terminal
tasks, and whether terminal state/history features are required for switching,
repair, verifier calls, and stop decisions.

The first fresh acquisition pass from the top-four BigCodeBench pool selected
8 unevaluated diverse tasks after excluding prior routing datasets and outcome
files. Qwen Coder solved 3/8 in a one-sample mining pass:
`BigCodeBench/123`, `BigCodeBench/678`, and `BigCodeBench/308`. A two-sample
repeat comparison across Qwen, Kimi, GLM, and DeepSeek produced three new
routing records, all with Qwen as the reward target. Merging these rows with the
11-task diagnostic dataset created a 14-task diagnostic set, but the temperature
selector quarantined every candidate: the best sweep reached 0.7143
leave-one-out target accuracy, 0.8571 pass@1, and 1334.1 ms mean latency regret.
Keep the rows as evidence, but do not promote a policy from this Qwen-heavy
refresh.

A follow-up non-Qwen mining pass over a second fresh 8-task top-four batch used
GLM, DeepSeek, and Kimi without Qwen. It found two specialist-positive
candidates, `BigCodeBench/1085` and `BigCodeBench/870`, but repeat comparison
with Qwen showed both are broad-pass rows where Qwen is much faster. Adding
them to the 14-task diagnostic set created a 16-task diagnostic set. The best
temperature candidates recovered to 0.75 leave-one-out target accuracy and
0.875 pass@1, but mean leave-one-out latency regret stayed at 1167.3 ms, above
the promotion gate. The next acquisition pass should search for tasks where
Qwen fails or is materially less reliable, not just tasks that specialists can
also solve.

The fresh-batch selector now has a `hard` strategy that favors higher
environment risk, higher plausibility score, and library/category novelty. Its
first hard top-four batch found the first fresh Qwen-negative Kimi targets:
`BigCodeBench/1004` and `BigCodeBench/1006`. Qwen failed both tasks 0/2 in the
repeat comparison, while Kimi passed both 2/2. The same run also produced
`BigCodeBench/760`, which remains a Qwen target because Qwen passed 2/2 and the
specialists were unstable. A 19-task diagnostic refresh is still quarantined:
the best low-temperature candidate has strong latency regret at 288.3 ms, but
leave-one-out target accuracy is 0.7368, just below the 0.75 gate. Treat this
as useful evidence and a feature-learning problem, not a promotion.

Network/archive/request interaction features resolved that promotion blocker.
The feature extractor now exposes compact signals such as
`combo_network_archive`, `combo_network_plotting`, and
`combo_network_filesystem`. With those features, the 19-task reward-tempered
router at temperature 0.05 passed the refresh gate and became active. Its
leave-one-out diagnostics are 0.7895 target accuracy, 0.8421 pass@1, and
606.6 ms mean latency regret. On the 19-task operational comparison, the active
logits router solves 17/19 with 0.8421 target accuracy, beating the fastest
single-worker baseline at 13/19 solved and the strongest-worker baseline at
14/19 solved.

A fresh held-out hard diagnostic then selected `BigCodeBench/100`,
`BigCodeBench/763`, `BigCodeBench/1022`, and `BigCodeBench/955`, excluding all
prior routing datasets and outcome files. With two samples per worker, only
`BigCodeBench/763` was solved: DeepSeek passed 2/2, Kimi passed 1/2, and Qwen
passed 0/2. The active 19-task router predicted Qwen for all four rows, so
held-out pass@1 was 0.0 even though target accuracy was 0.75 because the three
all-fail rows target the fastest failure path. Future promotion gates must
use solvable-row pass@1 separately from target accuracy and latency regret. The
active-policy and baseline reports now include `solvable_task_count`,
`solvable_pass_at_1`, and `solvable_target_accuracy`; refresh selection can now
enforce `min_loo_solvable_pass_at_1` when promoting candidate policies.

The held-out hard evidence was then merged into a 23-task candidate dataset and
swept with the solvable-row promotion gate enabled. Temperature 0.05 promoted:
LOO target accuracy is 0.7826, LOO pass@1 is 0.6957, LOO solvable pass@1 is
0.8000, and mean LOO latency regret is 501.1 ms. The active policy now points to
`research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`.
On the four-row hard slice it routes the only solvable task to DeepSeek, giving
1/4 pass@1 and 1.0 solvable pass@1. Treat that slice as a regression slice from
now on, not held-out evidence.

A subsequent fresh hard diagnostic selected `BigCodeBench/124`,
`BigCodeBench/526`, and `BigCodeBench/952`. Only `BigCodeBench/526` was
solvable: GLM passed 2/2, DeepSeek passed 1/2, and Qwen/Kimi passed 0/2. The
active 23-task router still predicted Qwen for all three rows, giving 0.0
pass@1 and 0.0 solvable pass@1 on the slice. This should be treated as a new
regression slice and a sign that the next improvement should be
solvability-aware decision structure, verifier/abstain behavior, or more
capacity, not just another immediate one-row promotion. Regression slices are
tracked in `research/evals/router_regression_slices.json` and evaluated with
`tools/evaluate_router_regression_slices.py`.

A first conditional-fallback evaluator now simulates verifier-guided retry over
measured outcomes. With `max_attempts=2`, it passes both regression slices: the
active logits router still fails the GLM `526` slice, but conditional fallback
rescues it by trying the next ranked worker after Qwen fails. On the active
23-task dataset, solved tasks improve from 18/23 to 19/23 and solvable pass@1
from 0.90 to 0.95, while mean latency rises from about 4.4s to 6.7s. This
supports a conditional verifier/abstain head as the next orchestration step.

A gated fallback variant reduces that latency cost. With `max_attempts=2` and a
top-vs-second router margin threshold of 0.10, it still solves 19/23 and keeps
solvable pass@1 at 0.95, but mean latency is about 4.9s instead of 6.7s. It
takes two fallbacks out of five first-failure opportunities and passes both
regression slices. This is the current best conditional workflow policy for
offline evaluation.

A live known-positive validation merge produced a 30-task repeated-routing
dataset and confirmed that promotion must guard both solvability and latency.
On the merged report, a frozen probe-gated policy keeps pass@1 at 0.7667 while
raising target accuracy to 0.8333 and reducing latency regret to 251.4 ms. Raw
live-augmented retrains reach up to 0.8000 LOO pass@1, above the strongest
single-worker baseline of 0.6667, but remain quarantined because they regress
target accuracy and latency regret. Future refreshes should compare against the
probe-gated result and use pass-first or conditional/probe-gated objectives
before attempting another active-policy promotion.

The refresh gate now supports that comparison directly through a named
operational-policy reference extracted from a benchmark report. The first
operational-reference sweep used the probe-gated policy as the preservation
target and quarantined all raw retrains. This should be the default benchmark
promotion pattern: compare candidates to the best measured operational policy,
not only to the previous active logits model.

The first gated-fallback candidate on the 30-task live-augmented dataset shows
the conditional-workflow tradeoff clearly. With margin 0.15 and two attempts,
it passes the regression slices and raises pass@1 from the probe-gated
reference's 0.7667 to 0.8333, but mean latency regret increases from 251.4 ms
to 1815.3 ms. Treat this as evidence for value-aware fallback selection rather
than promoting margin-only fallback.

A first learned second-attempt value policy was then trained without leaking the
second worker's pass result into its features. It also reaches pass@1 0.8333
and solvable pass@1 0.9259, but its selected threshold is still too permissive:
mean latency regret is 2087.4 ms and target accuracy falls to 0.7667. The next
benchmark acquisition pass should mine more fallback opportunities before using
the learned value head as an active workflow policy.

Historical fallback mining across the accumulated routing datasets produced 322
fallback-opportunity rows, but deduping leaves only 86 unique tasks and 16
positive fallback tasks. A mined fallback classifier reaches only 0.2222
leave-one-out F1, so this is not yet a reliable active fallback policy. Use the
corpus to choose the next benchmark acquisitions: novel fallback-positive
examples and cheap second-attempt rescues matter more than more duplicate
historical rows.

The corpus-guided acquisition selector now uses that mined fallback corpus as a
fresh-task prior. After excluding all measured routing rows and corpus-seen task
ids, it selected eight BigCodeBench-Hard candidates from 65 fresh normal-offset
eligible tasks:
`BigCodeBench/327`, `281`, `322`, `339`, `671`, `675`, `539`, and `673`.
These are mostly filesystem/text/general tasks with low Kimi/Qwen router
margins. Treat `research/evals/bigcodebench_hard_corpus_guided_fallback_batch8_tasks.json`
as the next live repeated top-worker batch, not as a benchmark result yet.

That batch has now been evaluated with two samples across the top-four cloud
workers. It produced six solvable tasks and two universal failures. GLM was
strongest by raw sample pass rate on the slice, while the latency-adjusted
targets were Qwen for `281`, GLM for `339`, and Kimi for `539`, `671`, `673`,
and `675`; the two universal failures target Qwen as the fastest failure path.
The active logits router solved every solvable row but had high latency regret,
and mining active-router top-fail cases produced only hard negatives. Use this
as latency-calibration and target-diversity evidence, not as a fallback-positive
value-head refresh.

The next acquisition selector is now rescue-pair-aware instead of only
low-margin or globally corpus-similar. It selects fresh tasks using historical
conditional evidence such as Qwen-to-DeepSeek, Qwen-to-GLM, Kimi-to-DeepSeek,
and Kimi-to-Qwen rescue pairs, while penalizing same-pair hard negatives. The
current selected batch is `BigCodeBench/283`, `365`, `125`, `672`, `565`, `644`,
`397`, and `370`. Treat it as the next repeated top-four evaluation target for
finding active-router top-fail alternate-pass labels.
