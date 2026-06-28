from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "active": None,
            "previous": None,
            "history": [],
        }
    return read_json(path)


def write_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_refresh(registry: dict[str, Any], refresh: dict[str, Any]) -> dict[str, Any]:
    if refresh["decision"] != "promote":
        raise ValueError(f"cannot apply refresh decision={refresh['decision']}")
    active = registry.get("active")
    candidate = refresh["candidate"]
    next_active = {
        "model": candidate["report"],
        "dataset": candidate["dataset"],
        "loo": candidate["loo"],
        "target_mix": candidate["target_mix"],
        "refresh_timestamp": refresh["timestamp"],
    }
    registry["previous"] = active
    registry["active"] = next_active
    registry.setdefault("history", []).append(
        {
            "action": "promote",
            "timestamp": utc_now(),
            "from": active,
            "to": next_active,
            "warnings": refresh.get("warnings", []),
        }
    )
    return registry


def rollback(registry: dict[str, Any]) -> dict[str, Any]:
    previous = registry.get("previous")
    if previous is None:
        raise ValueError("cannot rollback without previous active policy")
    active = registry.get("active")
    registry["active"] = previous
    registry["previous"] = active
    registry.setdefault("history", []).append(
        {
            "action": "rollback",
            "timestamp": utc_now(),
            "from": active,
            "to": previous,
        }
    )
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage active logits-router policy registry.")
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply_parser = subparsers.add_parser("apply-refresh")
    apply_parser.add_argument("--refresh", type=Path, required=True)

    subparsers.add_parser("rollback")

    args = parser.parse_args()
    registry = read_registry(args.registry)
    if args.command == "apply-refresh":
        registry = apply_refresh(registry, read_json(args.refresh))
    elif args.command == "rollback":
        registry = rollback(registry)
    else:
        raise ValueError(args.command)
    write_registry(args.registry, registry)
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
