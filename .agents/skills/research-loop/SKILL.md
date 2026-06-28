---
name: research-loop
description: Use when running bounded autonomous research experiments for mempool, including initializing runs, evaluating harness health, recording results, and progressing toward benchmark-backed orchestration improvements.
---

# Research Loop Skill

Use this skill when improving `mempool` through bounded autonomous experiments.

The loop is intentionally conservative: only run one experiment at a time, keep
the editable surface small, and record every result. The goal is steady,
auditable progress toward better orchestration policies.

## Setup

1. Pick a run tag, usually `YYYYMMDD-short-topic`.
2. Initialize the run:

```bash
python3 tools/research_loop.py init --tag <tag>
```

3. Read the active program:

```bash
sed -n '1,240p' research/programs/orchestration_loop.md
```

4. Read the benchmark plan:

```bash
sed -n '1,240p' docs/benchmark_strategy.md
cat research/evals/bigcodebench_hard_plan.json
```

5. For long-running orchestrator work, read the active roadmap:

```bash
sed -n '1,260p' research/programs/trainable_orchestrator_build_plan.md
cat research/programs/milestones.json
```

## Editable Surface

Allowed by default:

- `src/mempool/`
- `tests/`
- `tools/`
- `research/evals/`
- `research/programs/`
- `docs/benchmark_strategy.md`
- `docs/worker_pool.md`

Avoid editing public identity docs during experiments unless the user asks:

- `README.md`
- `docs/project_thesis.md`
- `AGENTS.md`

Do not edit reference checkouts in `external_repos/`.

## Evaluation

Run the fixed local evaluation:

```bash
python3 tools/research_loop.py evaluate --tag <tag>
```

This writes:

- `research/runs/<tag>/evaluations/*.json`
- `research/runs/<tag>/events.jsonl`

If a run changes what we believe, add a short note under `research/learnings/`.

The local score is only a harness health score until benchmark integration is
implemented. Treat it as a gate: tests and ledger behavior must remain healthy
before running any expensive external benchmark.

## Experiment Loop

For each experiment:

1. Record the starting commit or working-tree state.
2. Make one coherent change.
3. Run:

```bash
python3 tools/research_loop.py evaluate --tag <tag>
```

4. Append the result:

```bash
python3 tools/research_loop.py record --tag <tag> --status keep --description "<short description>"
```

Use `--status discard` for changes that fail to improve the selected metric, and
`--status crash` for broken experiments.

5. Keep only changes that improve the target metric or improve simplicity
without hurting the metric.

## Current Target

Build toward BigCodeBench-Hard instruct mode with this ladder:

1. 10-task smoke set
2. 30-50 task pilot
3. full hard subset

The first implementation target is not score maximization. It is a reliable
harness that can compare:

- best single strong worker
- cheapest acceptable worker
- strongest open/local worker
- rule router
- learned coordinator
- learned coordinator with conditional verifier
