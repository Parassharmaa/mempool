from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from .smoke_benchmark import SmokeCodeTask


DATASET_API = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "bigcode/bigcodebench-hard"
DEFAULT_SPLIT = "v0.1.4"


@dataclass(frozen=True)
class BigCodeBenchSource:
    dataset: str = DEFAULT_DATASET
    config: str = "default"
    split: str = DEFAULT_SPLIT
    offset: int = 0
    limit: int = 10


@dataclass(frozen=True)
class CanonicalProbe:
    task_id: str
    passed: bool
    score: float
    failure_mode: str | None
    stderr_tail: str = ""
    stdout_tail: str = ""


LIBRARY_CATEGORIES: dict[str, str] = {
    "asyncio": "concurrency",
    "collections": "algorithmic",
    "csv": "filesystem",
    "datetime": "datetime",
    "fnmatch": "filesystem",
    "glob": "filesystem",
    "gzip": "filesystem",
    "heapq": "algorithmic",
    "html": "text",
    "io": "filesystem",
    "itertools": "algorithmic",
    "json": "filesystem",
    "math": "general",
    "matplotlib": "plotting",
    "multiprocessing": "subprocess",
    "nltk": "nlp",
    "numpy": "datasci",
    "os": "filesystem",
    "pandas": "datasci",
    "pathlib": "filesystem",
    "pickle": "filesystem",
    "pytz": "datetime",
    "re": "text",
    "requests": "network",
    "scipy": "datasci",
    "seaborn": "plotting",
    "shutil": "filesystem",
    "sklearn": "datasci",
    "socket": "network",
    "ssl": "network",
    "statistics": "general",
    "string": "text",
    "subprocess": "subprocess",
    "tarfile": "filesystem",
    "tempfile": "filesystem",
    "threading": "concurrency",
    "time": "datetime",
    "urllib": "network",
    "zipfile": "filesystem",
}


def ensure_unittest_runner(test_source: str) -> str:
    if "unittest.main(" in test_source:
        return test_source
    if "unittest.TestCase" not in test_source and "unittest" not in test_source:
        return test_source
    return (
        test_source.rstrip()
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + "    unittest.main()\n"
    )


def canonical_output(row: dict[str, Any]) -> str:
    output = f"{row.get('code_prompt') or ''}{row.get('canonical_solution') or ''}"
    return output if output.endswith("\n") else output + "\n"


def normalize_bigcodebench_row(row: dict[str, Any], mode: str = "instruct") -> SmokeCodeTask:
    prompt_key = "instruct_prompt" if mode == "instruct" else "complete_prompt"
    prompt = str(row.get(prompt_key) or row.get("instruct_prompt") or row.get("complete_prompt") or "")
    libs = str(row.get("libs") or "").strip()
    entry_point = str(row.get("entry_point") or "task_func").strip()
    if libs:
        prompt = f"{prompt}\n\nYou may use these Python libraries if helpful: {libs}."
    prompt = f"{prompt}\n\nDefine `{entry_point}` exactly. Return only Python code."

    task_id = str(row["task_id"])
    safe_id = task_id.replace("/", "-")
    return SmokeCodeTask(
        id=f"bigcodebench-hard-{safe_id}",
        prompt=prompt,
        family="bigcodebench_hard",
        function_name=entry_point,
        tests=(ensure_unittest_runner(str(row.get("test") or "")),),
    )


def task_to_external_json(task: SmokeCodeTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "prompt": task.prompt,
        "family": task.family,
        "function_name": task.function_name,
        "tests": list(task.tests),
    }


def parse_prompt_libraries(prompt: str) -> list[str]:
    marker = "You may use these Python libraries if helpful:"
    if marker not in prompt:
        return []
    tail = prompt.split(marker, 1)[1].strip()
    candidate = tail.split(".", 1)[0].strip()
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        parsed = [part.strip(" '\"") for part in candidate.strip("[]").split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list | tuple):
        return []
    seen: set[str] = set()
    libraries: list[str] = []
    for item in parsed:
        library = str(item).strip()
        if library and library not in seen:
            seen.add(library)
            libraries.append(library)
    return libraries


def _library_available(library: str) -> bool:
    try:
        return importlib.util.find_spec(library.split(".", 1)[0]) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def classify_task(task: dict[str, Any]) -> dict[str, Any]:
    prompt = str(task.get("prompt") or "")
    prompt_lower = prompt.lower()
    libraries = parse_prompt_libraries(prompt)
    categories = {LIBRARY_CATEGORIES.get(library, "general") for library in libraries}
    if any(token in prompt_lower for token in ("file", "csv", "json", "directory", "path", "archive", "zip")):
        categories.add("filesystem")
    if any(token in prompt_lower for token in ("subprocess", "process", "execute", "command")):
        categories.add("subprocess")
    if any(token in prompt_lower for token in ("dataframe", "numpy", "pandas")):
        categories.add("datasci")
    if any(token in prompt_lower for token in ("plot", "chart", "matplotlib", "seaborn")):
        categories.add("plotting")
    if any(token in prompt_lower for token in ("socket", "http", "request", "url", "network")):
        categories.add("network")
    if any(token in prompt_lower for token in ("text", "regex", "string", "parse")):
        categories.add("text")
    ordered_categories = sorted(categories) or ["general"]
    missing_libraries = sorted(library for library in libraries if not _library_available(library))
    environment_risk = len(missing_libraries)
    environment_risk += sum(1 for category in ordered_categories if category in {"network", "subprocess"})
    environment_risk += sum(2 for category in ordered_categories if category in {"datasci", "image", "nlp", "plotting"})
    plausibility_score = float(len(libraries)) + float(environment_risk) * 1.5
    plausibility_score += float(sum(1 for test in task.get("tests", []) if len(str(test)) > 2500))
    return {
        "task_id": str(task.get("id") or task.get("task_id") or ""),
        "libraries": libraries,
        "categories": ordered_categories,
        "primary_category": ordered_categories[0],
        "missing_libraries": missing_libraries,
        "environment_risk": environment_risk,
        "plausibility_score": round(plausibility_score, 4),
    }


def probe_canonical_solution(row: dict[str, Any], adapter: Any, mode: str = "instruct") -> CanonicalProbe:
    task = normalize_bigcodebench_row(row, mode=mode)
    result = adapter.evaluate_output(task, canonical_output(row))
    return CanonicalProbe(
        task_id=task.id,
        passed=bool(result.passed),
        score=float(result.score),
        failure_mode=result.failure_mode,
        stderr_tail=str(result.metadata.get("stderr_tail", "")) if result.metadata else "",
        stdout_tail=str(result.metadata.get("stdout_tail", "")) if result.metadata else "",
    )


def scan_canonical_pass_rows(
    source: BigCodeBenchSource,
    adapter: Any,
    target_passes: int,
    page_size: int,
    max_rows: int,
    mode: str = "instruct",
) -> dict[str, Any]:
    passed_rows: list[dict[str, Any]] = []
    probes: list[CanonicalProbe] = []
    scanned = 0
    offset = source.offset
    while scanned < max_rows and len(passed_rows) < target_passes:
        limit = min(page_size, max_rows - scanned)
        rows = fetch_rows(BigCodeBenchSource(source.dataset, source.config, source.split, offset, limit))
        if not rows:
            break
        for row in rows:
            if scanned >= max_rows or len(passed_rows) >= target_passes:
                break
            probe = probe_canonical_solution(row, adapter=adapter, mode=mode)
            probes.append(probe)
            scanned += 1
            offset += 1
            if probe.passed:
                passed_rows.append(row)
        if len(rows) < limit:
            break
    return {"source": source, "scanned": scanned, "next_offset": offset, "passed_rows": passed_rows, "probes": probes}


def select_minipilot_tasks(
    tasks: list[dict[str, Any]],
    count: int,
    preferred_categories: tuple[str, ...] = ("subprocess", "filesystem", "datasci"),
    eligibility: dict[str, bool] | None = None,
) -> dict[str, Any]:
    eligibility = eligibility or {}
    analyses = [classify_task(task) for task in tasks]
    by_id = {str(task["id"]): task for task in tasks}
    analysis_by_id = {analysis["task_id"]: analysis for analysis in analyses}
    selected_ids: list[str] = []
    reasons: dict[str, str] = {}
    for category in preferred_categories:
        for analysis in analyses:
            task_id = analysis["task_id"]
            if task_id not in selected_ids and eligibility.get(task_id, True) and category in analysis["categories"]:
                selected_ids.append(task_id)
                reasons[task_id] = f"preferred_category:{category}"
                break
        if len(selected_ids) >= count:
            break
    for analysis in sorted(analyses, key=lambda item: (float(item["environment_risk"]), float(item["plausibility_score"]), item["task_id"])):
        task_id = analysis["task_id"]
        if len(selected_ids) >= count:
            break
        if task_id not in selected_ids and eligibility.get(task_id, True):
            selected_ids.append(task_id)
            reasons[task_id] = "lowest_risk_fill"
    return {
        "selected_task_ids": selected_ids,
        "selected_tasks": [by_id[task_id] for task_id in selected_ids],
        "selection_reasons": reasons,
        "analysis": [analysis_by_id[task_id] for task_id in selected_ids],
    }


def merge_task_lists(task_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for tasks in task_lists:
        for task in tasks:
            by_id.setdefault(str(task["id"]), task)
    return [by_id[task_id] for task_id in sorted(by_id)]


def merge_scan_reports(reports: list[dict[str, Any]], merged_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = [str(task["id"]) for task in merged_tasks]
    return {
        "eligible_count": len(merged_tasks),
        "scanned": sum(int(report.get("scanned", 0)) for report in reports),
        "next_offset": max((int(report.get("next_offset", 0)) for report in reports), default=0),
        "selected_task_ids": selected_ids,
        "analysis": [item for report in reports for item in report.get("analysis", []) if str(item.get("task_id")) in selected_ids],
        "canonical_probe": [item for report in reports for item in report.get("canonical_probe", []) if str(item.get("task_id")) in selected_ids],
    }


def load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "rows" in data:
        return [item.get("row", item) for item in data["rows"]]
    raise ValueError(f"unsupported BigCodeBench source shape: {path}")


def fetch_rows(source: BigCodeBenchSource) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "offset": source.offset,
            "length": source.limit,
        }
    )
    with urlopen(f"{DATASET_API}?{query}", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["row"] for item in payload.get("rows", [])]


def materialize_tasks(rows: list[dict[str, Any]], mode: str = "instruct") -> list[dict[str, Any]]:
    return [task_to_external_json(normalize_bigcodebench_row(row, mode=mode)) for row in rows]
