# First Signal - 2026-06-27

## Question

Can a simple router match the pass rate of an always-strong worker while reducing
cost on a tiny executable smoke benchmark?

## Setup

Command:

```bash
PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260627-first-signal
python3 tools/research_loop.py record --tag 20260627-first-signal --status keep --description "first smoke signal: rule router matches strong pass rate at lower cost"
```

Benchmark:

- `smoke-code`
- 4 local executable Python tasks
- fixture workers only
- not a model capability claim

## Result

| mode | solved | pass@1 | cost per solved task | mean latency ms |
| --- | ---: | ---: | ---: | ---: |
| cheap-baseline | 2/4 | 0.50 | 0.0020 | 65.25 |
| strong-fixture | 4/4 | 1.00 | 0.0400 | 140.00 |
| rule-router | 4/4 | 1.00 | 0.0205 | 100.50 |

## Interpretation

The first measurement spine works. On the smoke set, the rule router matches the
always-strong worker's pass rate while cutting cost per solved task roughly in
half. This is only a harness signal using fixture outputs, but it validates the
next step: replacing fixtures with real model adapters and running a 10-task
external smoke subset.
