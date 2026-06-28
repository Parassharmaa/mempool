# Evaluator Environment Provenance

Outcome rows now include evaluator Python provenance:

- `evaluator_python`
- `evaluator_python_version`
- `evaluator_required_packages`

The real-worker runner also supports `--required-package` checks. This turns the
catalog-candidate rerun lesson into a reusable guard: dependency-profile
BigCodeBench tasks should fail before model calls if the evaluator environment
does not have the expected packages.

Use this for top-four dependency tasks:

```bash
PYTHONPATH=src .venv-bigcodebench/bin/python tools/run_real_smoke_benchmark.py \
  --required-package numpy \
  --required-package pandas \
  --required-package matplotlib \
  --required-package requests \
  ...
```
