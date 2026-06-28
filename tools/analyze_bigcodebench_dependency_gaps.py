from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")

PACKAGE_BY_MODULE = {
    "cgi": "legacy-cgi",
    "docx": "python-docx",
    "flask_login": "flask-login",
    "flask_mail": "flask-mail",
    "Levenshtein": "python-Levenshtein",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
}


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_for_module(module: str) -> str:
    return PACKAGE_BY_MODULE.get(module, module)


def missing_module(stderr_tail: str) -> str | None:
    match = MISSING_MODULE_RE.search(stderr_tail)
    return match.group(1) if match else None


def analyze_reports(paths: list[Path]) -> dict[str, Any]:
    by_package: dict[str, list[dict[str, Any]]] = defaultdict(list)
    non_dependency_failures = []
    scanned = 0
    eligible = 0
    for path in paths:
        report = read_report(path)
        scanned += int(report.get("scanned", 0))
        eligible += int(report.get("eligible_count", 0))
        for probe in report.get("canonical_probe", []):
            if probe.get("passed"):
                continue
            module = missing_module(str(probe.get("stderr_tail") or ""))
            if not module:
                non_dependency_failures.append(
                    {
                        "report": str(path),
                        "task_id": probe.get("task_id"),
                        "failure_mode": probe.get("failure_mode"),
                        "stderr_tail": probe.get("stderr_tail", "")[-500:],
                    }
                )
                continue
            package = package_for_module(module)
            by_package[package].append(
                {
                    "report": str(path),
                    "task_id": probe.get("task_id"),
                    "module": module,
                }
            )

    package_counts = Counter({package: len(tasks) for package, tasks in by_package.items()})
    ranked_packages = [
        {
            "package": package,
            "blocked_task_count": count,
            "modules": sorted({task["module"] for task in by_package[package]}),
            "task_ids": sorted({task["task_id"] for task in by_package[package]}),
        }
        for package, count in package_counts.most_common()
    ]
    cumulative = []
    unlocked = set()
    for item in ranked_packages:
        unlocked.update(item["task_ids"])
        cumulative.append(
            {
                "packages": [entry["package"] for entry in ranked_packages[: len(cumulative) + 1]],
                "unique_blocked_tasks": len(unlocked),
            }
        )

    return {
        "reports": [str(path) for path in paths],
        "scanned": scanned,
        "eligible": eligible,
        "dependency_blocked_tasks": sum(item["blocked_task_count"] for item in ranked_packages),
        "unique_dependency_blocked_tasks": len(
            {task["task_id"] for tasks in by_package.values() for task in tasks}
        ),
        "ranked_packages": ranked_packages,
        "cumulative_unlock": cumulative,
        "non_dependency_failure_count": len(non_dependency_failures),
        "non_dependency_failures": non_dependency_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze BigCodeBench canonical failures caused by missing dependencies."
    )
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_reports(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
