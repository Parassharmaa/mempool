from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .adapters import OpenAICompatibleClient, OpenAICompatibleConfig
from .orchestrator_runtime import build_prompt_record, predict_orchestration


class ChatClient(Protocol):
    def chat(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_env_file(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    loaded = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        loaded.append(key)
    return loaded


def load_worker_pool(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("base_url"):
        raise ValueError("worker pool missing base_url")
    workers = payload.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("worker pool needs a non-empty workers list")
    for index, worker in enumerate(workers):
        if not worker.get("id") or not worker.get("model"):
            raise ValueError(f"worker {index} needs id and model")
    return payload


def worker_by_id(worker_pool: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker in worker_pool["workers"]:
        if worker["id"] == worker_id:
            return worker
    raise ValueError(f"selected worker {worker_id!r} is not present in worker pool")


def chat_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def build_openai_compatible_client(worker_pool: dict[str, Any]) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        OpenAICompatibleConfig(
            base_url=worker_pool["base_url"],
            api_key_env=worker_pool.get("api_key_env"),
            timeout_seconds=int(worker_pool.get("timeout_seconds", 120)),
            chat_options=worker_pool.get("chat_options"),
        )
    )


def execute_orchestrated_prompt(
    *,
    model_path: str | Path,
    worker_pool_path: str | Path,
    prompt: str,
    task_id: str = "ad-hoc",
    benchmark_id: str = "ad-hoc",
    task_family: str = "ad_hoc",
    categories: list[str] | None = None,
    libraries: list[str] | None = None,
    missing_libraries: list[str] | None = None,
    system_prompt: str = "You are a helpful worker model. Return the best answer you can.",
    client: ChatClient | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    worker_pool = load_worker_pool(worker_pool_path)
    record = build_prompt_record(
        prompt=prompt,
        task_id=task_id,
        benchmark_id=benchmark_id,
        task_family=task_family,
        categories=categories,
        libraries=libraries,
        missing_libraries=missing_libraries,
    )
    route = predict_orchestration(model_path=model_path, record=record)
    selected_worker = worker_by_id(worker_pool, route["selected_worker_id"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if dry_run:
        return {
            "schema_version": "mempool.orchestrated_execution.v1",
            "timestamp": utc_now(),
            "model_path": str(model_path),
            "worker_pool_path": str(worker_pool_path),
            "task_id": task_id,
            "benchmark_id": benchmark_id,
            "task_family": task_family,
            "prompt": prompt,
            "route": route,
            "selected_worker": {
                "id": selected_worker["id"],
                "model": selected_worker["model"],
                "strengths": selected_worker.get("strengths", []),
                "cost_usd": selected_worker.get("cost_usd", 0.0),
            },
            "request": {
                "messages": messages,
                "chat_options": worker_pool.get("chat_options") or {},
            },
            "response": {
                "content": "",
                "raw": {},
            },
            "latency_ms": 0,
            "execution_status": "dry_run",
        }
    chat_client = client or build_openai_compatible_client(worker_pool)
    started = time.perf_counter()
    response = chat_client.chat(selected_worker["model"], messages)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "schema_version": "mempool.orchestrated_execution.v1",
        "timestamp": utc_now(),
        "model_path": str(model_path),
        "worker_pool_path": str(worker_pool_path),
        "task_id": task_id,
        "benchmark_id": benchmark_id,
        "task_family": task_family,
        "prompt": prompt,
        "route": route,
        "selected_worker": {
            "id": selected_worker["id"],
            "model": selected_worker["model"],
            "strengths": selected_worker.get("strengths", []),
            "cost_usd": selected_worker.get("cost_usd", 0.0),
        },
        "request": {
            "messages": messages,
            "chat_options": worker_pool.get("chat_options") or {},
        },
        "response": {
            "content": chat_content(response),
            "raw": response,
        },
        "latency_ms": latency_ms,
        "execution_status": "completed",
    }


def flatten_orchestrated_execution(result: dict[str, Any]) -> dict[str, Any]:
    route = result.get("route") or {}
    selected_worker = result.get("selected_worker") or {}
    response = result.get("response") or {}
    content = response.get("content")
    execution_status = str(result.get("execution_status") or "unknown")
    return {
        "schema_version": "mempool.orchestrated_execution_outcome.v1",
        "benchmark_id": result.get("benchmark_id"),
        "task_id": result.get("task_id"),
        "task_family": result.get("task_family"),
        "prompt": result.get("prompt"),
        "timestamp": result.get("timestamp"),
        "model_path": result.get("model_path"),
        "worker_pool_path": result.get("worker_pool_path"),
        "selected_worker_id": selected_worker.get("id"),
        "selected_model": selected_worker.get("model"),
        "selected_workflow": route.get("selected_workflow"),
        "worker_distribution": route.get("worker_distribution", {}),
        "workflow_distribution": route.get("workflow_distribution", {}),
        "verifier_probability": route.get("verifier_probability"),
        "abstain_probability": route.get("abstain_probability"),
        "execution_status": execution_status,
        "latency_ms": result.get("latency_ms"),
        "cost_usd": selected_worker.get("cost_usd", 0.0),
        "response_chars": len(content) if isinstance(content, str) else 0,
        "response_present": bool(content),
        "passed": None,
        "score": None,
        "failure_mode": None if execution_status in {"completed", "dry_run"} else execution_status,
        "reward": None,
    }


def write_orchestrated_outcome(path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    row = flatten_orchestrated_execution(result)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return row
