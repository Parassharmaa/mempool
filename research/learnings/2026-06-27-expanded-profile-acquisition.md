# Expanded Profile Acquisition

## Question

Can a broader BigCodeBench evaluator dependency profile unlock fresh stable
routing rows after the standard top-four profile was nearly exhausted?

## Result

Yes. Adding the top dependency unblockers to the isolated BigCodeBench evaluator
environment converted a sparse first slice into a useful acquisition source.

Installed into `.venv-bigcodebench`:

- `scikit-learn` (`sklearn` import)
- `scipy`
- `seaborn`
- `beautifulsoup4` (`bs4` import)

The tracked evaluator profile now records import-module names because the
runner and conversion guards check importability, not package metadata:

- `numpy`
- `pandas`
- `matplotlib`
- `requests`
- `sklearn`
- `scipy`
- `seaborn`
- `bs4`

An expanded-profile canonical scan over the first 30 BigCodeBench-Hard rows
found 19 locally runnable canonical-pass tasks. After excluding active routing
data and every prior raw outcome, 6 fresh tasks remained:

- `BigCodeBench/184`
- `BigCodeBench/92`
- `BigCodeBench/129`
- `BigCodeBench/93`
- `BigCodeBench/37`
- `BigCodeBench/99`

The repeated worker comparison produced 48 outcome rows across Qwen, Kimi, GLM,
and DeepSeek. The outcome audit passed with all required evaluator modules
present.

Conversion yielded 6 routing records. Merge filtering dropped
`BigCodeBench/93` as all-fail and kept 5 stable rows.

## Router Decision

Merging these rows with the prior 33-task candidate produced a 38-task candidate
dataset with this target mix:

- Qwen: 21
- Kimi: 7
- DeepSeek: 7
- GLM: 3

The candidate was quarantined by the refresh gate:

- LOO target accuracy fell from 0.783 to 0.474.
- LOO solvable pass@1 stayed at 0.800.
- LOO mean latency regret rose from 501 ms to 4430 ms.

The new data is useful, but the current linear logits router does not yet
generalize across the added datasci, plotting, and network rows. The active
23-task policy should remain unchanged.

## Decision

Keep the expanded-profile rows as measured evidence and continue acquiring from
this profile. Before promoting another larger dataset, improve the router's
feature/capacity path for dependency-heavy tasks and latency regret. The next
practical step is to scan additional expanded-profile offsets and collect enough
stable rows to reach the 50-row M5 data threshold, while treating candidate
routers as quarantined until leave-one-out latency regret is controlled.
