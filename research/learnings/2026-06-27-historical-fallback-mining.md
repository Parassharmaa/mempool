# Historical Fallback Mining

Run tag: `20260627-historical-fallback-mining`

## What Changed

Added `tools/mine_historical_fallback_cases.py` to mine existing routing datasets
under the active logits router. The miner emits only active-router fallback
opportunities: records where the router's top worker failed and at least one
alternate could be considered.

Each emitted row includes:

- top and second worker probabilities
- whether the second worker is a useful rescue
- whether any later-ranked alternate is a useful rescue
- best ranked passing alternate
- fastest passing alternate
- total and additional fallback latency
- source dataset provenance

## Result

Mining the active, held-out, post-refresh, and recent fallback-screen routing
datasets produced:

- 19 fallback opportunities
- 3 useful any-alternate fallbacks
- 2 useful second-worker fallbacks
- 16 hard negatives
- 16 unique tasks

The useful rescue tasks were:

- `bigcodebench-hard-BigCodeBench-368`: Qwen top failed; DeepSeek passed at rank 3.
- `bigcodebench-hard-BigCodeBench-526`: Qwen top failed; DeepSeek passed at rank 2.
- `bigcodebench-hard-BigCodeBench-963`: Qwen top failed; GLM passed at rank 2.

Fresh fallback acquisition by uncertainty or positive-neighborhood similarity had
produced many all-fail rows. Historical mining recovered the sparse positive
fallback labels without more cloud calls.

## Learning

Fallback action learning should not train only on fresh uncertain tasks. The
useful signal is sparse and appears in already collected specialist slices.
Before spending more worker calls, mine existing routing datasets for
top-fail/alternate-pass cases and use those rows as the first supervised
fallback-action dataset.

## Next Step

Train or select the fallback action head on the mined fallback cases, using the
hard negatives from recent screens as calibration pressure and the 368/526/963
positives as rescue supervision. Evaluate against the active 23-task set and
both regression slices before considering promotion.
