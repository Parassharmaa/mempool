# Datasets

The training data in this repository is measured local orchestration data, not a
public leaderboard scrape.

Most important current files:

- `20260628-m5-current-task-66task-routing.jsonl`: task-level routing records
  with repeated worker outcomes summarized into reward targets.
- `20260628-m5-current-task-66task-substrate.jsonl`: supervised multi-head
  orchestrator examples derived from the routing records.
- `20260628-m5-current-task-66task-substrate-manifest.json`: manifest for the
  66-record substrate.
- `20260628-fallback-opportunity-corpus.jsonl`: mined active-router
  top-fail/alternate-pass fallback cases.

The current trained orchestrator checkpoint is:

- `../models/20260628-m5-current-task-66task-multihead.json`

## Hugging Face Upload

Hugging Face auth was not active in the local environment when this repo was
prepared. Once authenticated, upload the dataset subset with:

```bash
hf auth login
hf repo create mempool-orchestration-data --type dataset --public
hf upload Parassharmaa/mempool-orchestration-data research/datasets . \
  --repo-type dataset \
  --exclude "*.pyc" \
  --exclude "__pycache__/*"
```

If publishing broadly, keep raw provider outputs, secrets, local logs, and full
terminal transcripts out of the dataset repository.
