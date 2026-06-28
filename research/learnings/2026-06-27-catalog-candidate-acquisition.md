# Catalog Candidate Acquisition

The refreshed catalog candidates were tested on two known regression slices:

- `BigCodeBench/763`, where DeepSeek is the stable target.
- `BigCodeBench/526`, where GLM is the stable target.

The first run used base Python and produced invalid failures because these
top-four dependency tasks need `numpy` and `pandas`. The valid rerun used
`.venv-bigcodebench`.

Valid rerun result:

- `ollama-cloud-qwen3-coder-next`: 1/4 samples overall, with 1/2 on `763` and
  0/2 on `526`; mean latency 3517 ms.
- `ollama-cloud-qwen3p5-397b`: 0/4; mean latency 47812 ms.
- `ollama-cloud-gpt-oss-120b`: 0/4; mean latency 5802 ms.

Conclusion: do not merge these candidate rows into the active router dataset
yet. The strongest useful signal is that `qwen3-coder-next` may deserve a
broader low-cost coding smoke, but it does not replace the measured DeepSeek or
GLM specialists on these regression slices.
