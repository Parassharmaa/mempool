# Harness Source Repair

Run tag: `20260628-harness-source-repair`

## Question

Can the repo recover cleanly after the BigCodeBench evaluator `cwd` bug deleted
`src/mempool`, and is the local research-loop gate healthy enough to continue
benchmark acquisition?

## Repairs

- Restored missing source APIs required by the current unit suite.
- Kept the smoke evaluator isolation fix: candidate code now runs with
  `cwd` set to the temporary candidate directory, not the repository root.
- Restored the demo module entrypoint so `python -m mempool.demo` writes a
  `workflow_planned` ledger event.
- Relabeled `smoke-parse-kv` from `code_text` to `code_data`, because it is a
  dict-construction parsing task and the rule router should send it to the
  stronger data-capable fixture.
- Restored `kw_file`/`kw_files` task features so fallback-opportunity selection
  can distinguish file/archive tasks from unrelated low-margin tasks.

## Evaluation

Local unit suite:

- `PYTHONPATH=src python3 -m unittest discover -s tests`
- result: `220` tests passed

Research-loop gate:

- `PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260628-harness-source-repair`
- result: `pass`
- score: `1.0`
- checks: unit tests, demo run, ledger written, ledger parseable, smoke signal

The result was recorded as `keep`.

## Acquisition Follow-Up

After the gate recovered, the next local canonical eligibility scan was tried
from offset `676`:

- `PYTHONPATH=src python3 tools/scan_bigcodebench_eligible.py --offset 676 --target-passes 8 --page-size 10 --max-rows 80 --eval-timeout-seconds 20 --output research/evals/20260628-normal-offset676-eligible-tasks.json --report research/evals/20260628-normal-offset676-eligible-report.json`
- scanned rows: `0`
- eligible tasks: `0`
- next offset: `676`

This suggests the current BigCodeBench-Hard source path is exhausted at offset
`676`; the next acquisition step should not keep incrementing this normal
offset. Use an alternate source profile, dependency-profile expansion, or a
targeted non-Qwen specialist-pressure pool instead.

## Decision

Keep the harness repair. Do not run paid worker calls until the next candidate
source is selected from a non-exhausted pool.
