from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKAGE_ALIASES = {
    "cgi": "legacy-cgi",
    "docx": "python-docx",
    "flask_login": "flask-login",
    "flask_mail": "flask-mail",
    "Levenshtein": "python-Levenshtein",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
}


def normalize_package(package: str) -> str:
    value = package.strip()
    return PACKAGE_ALIASES.get(value, value)


def read_gap_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_profile(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    packages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            packages.append(normalize_package(value))
    return packages


def task_ids_for_package(gap_report: dict[str, Any], package: str) -> set[str]:
    for item in gap_report.get("ranked_packages", []):
        if item.get("package") == package:
            return {str(task_id) for task_id in item.get("task_ids", [])}
    return set()


def plan_expansion(
    gap_report: dict[str, Any],
    *,
    current_packages: list[str] | None = None,
    package_limit: int = 4,
) -> dict[str, Any]:
    if package_limit < 1:
        raise ValueError("package_limit must be at least 1")
    current_packages = current_packages or []
    current_packages = list(dict.fromkeys(normalize_package(package) for package in current_packages))
    current_set = set(current_packages)
    selected: list[dict[str, Any]] = []
    unlocked: set[str] = set()

    for item in gap_report.get("ranked_packages", []):
        package = normalize_package(str(item["package"]))
        if package in current_set:
            continue
        task_ids = {str(task_id) for task_id in item.get("task_ids", [])}
        incremental = task_ids - unlocked
        selected.append(
            {
                "package": package,
                "blocked_task_count": int(item.get("blocked_task_count", len(task_ids))),
                "modules": list(item.get("modules", [])),
                "task_ids": sorted(task_ids),
                "incremental_task_ids": sorted(incremental),
                "incremental_unique_tasks": len(incremental),
            }
        )
        unlocked.update(task_ids)
        if len(selected) >= package_limit:
            break

    expanded_profile = list(dict.fromkeys([*current_packages, *[item["package"] for item in selected]]))
    return {
        "current_profile": current_packages,
        "selected_packages": selected,
        "expanded_profile": expanded_profile,
        "selected_package_count": len(selected),
        "projected_unique_unlocks": len(unlocked),
        "source_unique_dependency_blocked_tasks": int(
            gap_report.get("unique_dependency_blocked_tasks", 0) or 0
        ),
        "source_scanned_tasks": int(gap_report.get("scanned", 0) or 0),
        "source_eligible_tasks": int(gap_report.get("eligible", 0) or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan the next isolated BigCodeBench dependency-profile expansion."
    )
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--current-profile", type=Path)
    parser.add_argument("--package-limit", type=int, default=4)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    args = parser.parse_args()

    gap_report = read_gap_report(args.gap_report)
    current_packages = read_profile(args.current_profile)
    plan = plan_expansion(
        gap_report,
        current_packages=current_packages,
        package_limit=args.package_limit,
    )
    plan["gap_report"] = str(args.gap_report)
    if args.current_profile:
        plan["current_profile_path"] = str(args.current_profile)
    plan["profile_output"] = str(args.profile_output)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.profile_output.write_text(
        "\n".join(plan["expanded_profile"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
