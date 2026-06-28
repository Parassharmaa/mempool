# AGENTS.md

## Project Identity

This project must remain independent in its public framing. Do not describe it
as a clone, open version, wrapper, or derivative of any external system.

Use outside systems only as literature, comparison points, or implementation
references. Keep names of external projects out of the README, project thesis,
roadmap, architecture overview, and paper drafts unless the file is explicitly a
literature review or citation document.

## Working Principles

- Read the relevant docs and code before changing behavior.
- Keep the core system small, inspectable, and testable.
- Prefer explicit schemas, structured logs, and reproducible experiments.
- Make routing decisions explainable after the fact.
- Treat autonomous experiments as bounded loops with fixed metrics and reviewable
  diffs.
- Preserve provenance for model calls, prompts, task metadata, verifier results,
  and final synthesis.

## Directory Guide

- `src/`: production implementation.
- `tools/`: scripts and local automation.
- `docs/`: project-facing architecture and plans. Keep this independent.
- `research/`: active ideas, experiment plans, results, and logs.
- `literature/`: summaries of papers and outside systems. External names are OK
  here when needed for citation and comparison.
- `external_repos/`: checked-out reference projects. Do not edit these directly
  unless the task explicitly says to modify a reference checkout.

## Development Workflow

No build system exists yet. When one is added, document exact commands here for:

```bash
# install

# test

# lint

# run a local experiment
```

Until then, changes should be mostly Markdown, schemas, and small scripts. Keep
new files ASCII unless a source file already uses another character set.

## Research Hygiene

- Distinguish confirmed evidence from hypotheses.
- Record negative results; they are part of the search space.
- Do not optimize only for answer quality. Track latency, cost, robustness,
  provenance, and failure recovery.
- Prefer small controlled evals before broad claims.
- Avoid hidden global state in experiments.

## Research Loop

For bounded autonomous improvement work, use
`.agents/skills/research-loop/SKILL.md`. Initialize a run, make one coherent change at a
time, run `python3 tools/research_loop.py evaluate --tag <tag>`, and record the
result. Do not edit `external_repos/` as part of the loop.
