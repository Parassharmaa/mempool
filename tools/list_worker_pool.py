from __future__ import annotations

import argparse
import json

from mempool.adapters import OpenAICompatibleClient, OpenAICompatibleConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="List models from a worker endpoint.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key-env")
    args = parser.parse_args()

    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=10,
        )
    )
    print(json.dumps({"models": list(client.list_models())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
