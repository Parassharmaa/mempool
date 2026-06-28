from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import BenchmarkResult, BenchmarkTask


@dataclass(frozen=True)
class SmokeCodeTask(BenchmarkTask):
    function_name: str = ""
    tests: tuple[str, ...] = ()


def extract_python_code(text: str) -> str:
    if "```" not in text:
        return text if text.endswith("\n") else text + "\n"
    parts = text.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index]
        if block.startswith("python\n"):
            code = block.removeprefix("python\n")
            return code if code.endswith("\n") else code + "\n"
        if block.startswith("py\n"):
            code = block.removeprefix("py\n")
            return code if code.endswith("\n") else code + "\n"
        if "\n" in block:
            code = block.split("\n", 1)[1]
            return code if code.endswith("\n") else code + "\n"
    return text if text.endswith("\n") else text + "\n"


class SmokeCodeBenchmarkAdapter:
    id = "smoke-code"

    def __init__(self, path: str | Path, timeout_seconds: int = 5) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds

    def load_tasks(self, limit: int | None = None) -> tuple[SmokeCodeTask, ...]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        tasks = tuple(
            SmokeCodeTask(
                id=item["id"],
                prompt=item["prompt"],
                family=item["family"],
                function_name=item["function_name"],
                tests=tuple(item["tests"]),
            )
            for item in data
        )
        return tasks[:limit] if limit else tasks

    def evaluate_output(self, task: BenchmarkTask, output: str) -> BenchmarkResult:
        if not isinstance(task, SmokeCodeTask):
            return BenchmarkResult(
                task_id=task.id,
                passed=False,
                score=0.0,
                failure_mode="wrong_task_type",
            )

        test_source = output + "\n\n" + "\n".join(task.tests) + "\n"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "candidate.py"
                path.write_text(test_source, encoding="utf-8")
                completed = subprocess.run(
                    [sys.executable, str(path)],
                    cwd=tmpdir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            return BenchmarkResult(
                task_id=task.id,
                passed=False,
                score=0.0,
                failure_mode="eval_timeout",
                metadata={
                    "timeout_seconds": self.timeout_seconds,
                    "stderr_tail": (exc.stderr or "")[-500:] if isinstance(exc.stderr, str) else "",
                    "stdout_tail": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "",
                },
            )

        passed = completed.returncode == 0
        return BenchmarkResult(
            task_id=task.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            failure_mode=None if passed else "test_failure",
            metadata={
                "stderr_tail": completed.stderr[-500:],
                "stdout_tail": completed.stdout[-500:],
                "timeout_seconds": self.timeout_seconds,
            },
        )


FIXTURE_OUTPUTS: dict[str, dict[str, str]] = {
    "cheap-baseline": {
        "smoke-add-numbers": "def add_numbers(a, b):\n    return a + b\n",
        "smoke-reverse-words": "def reverse_words(text):\n    return ' '.join(reversed(text.split()))\n",
        "smoke-normalize-records": "def normalize_records(records):\n    return records\n",
        "smoke-top-k-frequent": "def top_k_frequent(items, k):\n    return items[:k]\n",
        "smoke-flatten-list": (
            "def flatten_once(items):\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(item)\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        "smoke-safe-int": (
            "def safe_int(value, default=0):\n"
            "    try:\n"
            "        return int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default\n"
        ),
        "smoke-title-case": "def title_case_words(text):\n    return ' '.join(text.lower().split()).title()\n",
    },
    "strong-fixture": {
        "smoke-add-numbers": "def add_numbers(a, b):\n    return a + b\n",
        "smoke-reverse-words": "def reverse_words(text):\n    return ' '.join(reversed(text.split()))\n",
        "smoke-normalize-records": (
            "def normalize_records(records):\n"
            "    out = []\n"
            "    for record in records:\n"
            "        out.append({key.lower(): value.strip() if isinstance(value, str) else value for key, value in record.items()})\n"
            "    return out\n"
        ),
        "smoke-top-k-frequent": (
            "from collections import Counter\n\n"
            "def top_k_frequent(items, k):\n"
            "    counts = Counter(items)\n"
            "    return [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:k]]\n"
        ),
        "smoke-flatten-list": (
            "def flatten_once(items):\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(item)\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        "smoke-safe-int": (
            "def safe_int(value, default=0):\n"
            "    try:\n"
            "        return int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default\n"
        ),
        "smoke-group-by-key": (
            "def group_by_key(records, key):\n"
            "    out = {}\n"
            "    for record in records:\n"
            "        out.setdefault(record.get(key), []).append(record)\n"
            "    return out\n"
        ),
        "smoke-title-case": "def title_case_words(text):\n    return ' '.join(text.lower().split()).title()\n",
        "smoke-windowed": (
            "def windowed(items, size):\n"
            "    if size < 1:\n"
            "        return []\n"
            "    return [tuple(items[index:index + size]) for index in range(0, len(items) - size + 1)]\n"
        ),
        "smoke-parse-kv": (
            "def parse_kv_pairs(text):\n"
            "    out = {}\n"
            "    for part in text.split(','):\n"
            "        if '=' not in part:\n"
            "            continue\n"
            "        key, value = part.split('=', 1)\n"
            "        key = key.strip()\n"
            "        if key:\n"
            "            out[key] = value.strip()\n"
            "    return out\n"
        ),
    },
}


def fixture_output(worker_id: str, task_id: str) -> str:
    return FIXTURE_OUTPUTS.get(worker_id, {}).get(task_id, "")


def task_to_dict(task: SmokeCodeTask) -> dict[str, object]:
    data = asdict(task)
    data["tests"] = list(task.tests)
    return data
