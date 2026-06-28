# Fallback Opportunity Corpus

Run tag: `20260628-fallback-opportunity-corpus`

## Question

Can existing measured routing data provide enough fallback-opportunity examples
to train a better value-aware fallback signal?

## Change

- Patched `tools/mine_historical_fallback_cases.py` to ignore router-ranked
  workers that were not measured in a historical dataset row.
- Removed the outcome-derived `passed_alternate_count` feature from the mined
  fallback head.
- Mined all current `*routing.jsonl` datasets into
  `research/datasets/20260628-fallback-opportunity-corpus.jsonl`.
- Trained a mined fallback logit head on the deduped corpus.

## Corpus

The mined corpus contains:

- 322 fallback-opportunity rows
- 86 unique tasks after dedupe
- 149 useful-any fallback rows before dedupe
- 78 useful-second fallback rows before dedupe
- 173 hard negatives before dedupe

Most active-router top failures are Qwen-first failures:

- Qwen top failures: 255
- Kimi top failures: 66
- DeepSeek top failures: 1

Positive alternates are distributed across specialists:

- DeepSeek: 54
- Kimi: 47
- GLM: 39
- Qwen: 9

## Model Result

After deduping by task id, the mined-head training set has only 86 rows and 16
positive tasks.

Training metrics:

- accuracy 0.8488
- precision 0.6000
- recall 0.5625
- F1 0.5806

Leave-one-out metrics:

- accuracy 0.7558
- precision 0.2727
- recall 0.1875
- F1 0.2222

The low leave-one-out F1 means the current mined head is not ready as a
promotion-grade fallback signal.

## Decision

Keep:

- fallback-opportunity corpus
- non-leaky mined fallback features
- mined fallback head artifact as a diagnostic model

Do not promote:

- mined fallback head as an active workflow policy

## Next

Use the corpus to guide targeted acquisition instead of immediate promotion.
The next data run should prioritize novel fallback-positive tasks, especially
cheap specialist rescues, and avoid simply duplicating the same task ids across
merged historical datasets.
