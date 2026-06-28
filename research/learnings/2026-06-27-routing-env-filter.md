# Routing Environment Filter

Evaluator provenance now affects dataset conversion, not only acquisition.

`tools/build_repeated_routing_dataset.py` and `tools/build_routing_dataset.py`
can require evaluator packages with `--required-evaluator-package`. Rows that
lack matching `evaluator_required_packages` are skipped before rewards and soft
targets are computed.

This is important for top-four BigCodeBench tasks: a base-Python run can create
plausible-looking outcome rows whose failures are really missing `numpy` or
`pandas`. Those rows must not be allowed to train the router.
