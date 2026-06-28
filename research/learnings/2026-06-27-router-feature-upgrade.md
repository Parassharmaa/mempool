# Router Feature Upgrade

## Question

Can dependency-aware feature cleanup or simple interaction features make the
38-task expanded-profile router generalize well enough to promote?

## Result

The data provenance fix is useful, but the router should not be promoted.

The repeated-routing converter previously derived `missing_libraries` by calling
`find_spec` in the converter process. That can be wrong when outcome rows were
evaluated in `.venv-bigcodebench` but converted from the project Python. The
converter now uses each outcome row's `evaluator_required_packages` map when it
is present, and falls back to local import detection for libraries that were not
part of the evaluator package contract. This fixed false missing-library labels
on the expanded-profile rows.

An attempted category-library interaction feature set was tested and discarded.
It improved in-sample fit but hurt leave-one-out behavior on the 38-task
dataset, suggesting overfit under the current data volume.

## Router Decision

The final provenance-fixed 38-task candidate was quarantined:

- LOO target accuracy: 0.474.
- LOO solvable pass@1: 0.771.
- LOO mean latency regret: 4409 ms.

Against the active 23-task baseline, the candidate misses the minimum target
accuracy, solvable pass@1, and latency-regret gates. The active 23-task policy
should remain unchanged.

## Decision

Keep the provenance correction in the dataset converter. Do not keep the
interaction-feature experiment in the active feature extractor. Continue toward
50 stable rows with the expanded dependency profile, then revisit model
capacity with stronger validation once there are enough examples for the new
datasci, plotting, and network regions.
