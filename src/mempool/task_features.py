from __future__ import annotations

import re
from typing import Any


KEYWORDS = (
    "archive",
    "base64",
    "count",
    "dict",
    "download",
    "fetch",
    "file",
    "files",
    "frequent",
    "hex",
    "http",
    "key",
    "list",
    "lowercase",
    "number",
    "plot",
    "request",
    "requests",
    "scan",
    "sorted",
    "string",
    "strip",
    "subprocess",
    "sum",
    "url",
    "word",
    "zip",
)

CATEGORIES = (
    "algorithmic",
    "concurrency",
    "datasci",
    "datetime",
    "filesystem",
    "general",
    "image",
    "network",
    "nlp",
    "plotting",
    "subprocess",
    "text",
)


def feature_safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def extract_task_features(record: dict[str, Any]) -> dict[str, float]:
    prompt = str(record.get("prompt", "")).lower()
    tokens = re.findall(r"[a-z0-9_]+", prompt)
    token_set = set(tokens)
    task_family = str(record.get("task_family", ""))
    prompt_features = record.get("prompt_features") or {}
    categories = [str(value) for value in prompt_features.get("categories", [])]
    libraries = [str(value) for value in prompt_features.get("libraries", [])]
    missing_libraries = [str(value) for value in prompt_features.get("missing_libraries", [])]

    features: dict[str, float] = {
        "bias": 1.0,
        "length_chars": float(len(prompt)),
        "length_tokens": float(len(tokens)),
    }
    if task_family:
        features[f"family_{feature_safe_name(task_family)}"] = 1.0

    for keyword in KEYWORDS:
        features[f"kw_{keyword}"] = float(keyword in token_set)

    for category in categories:
        safe = feature_safe_name(category)
        if safe:
            features[f"category_{safe}"] = 1.0
            features[f"signal_{safe}"] = 1.0

    for library in libraries:
        safe = feature_safe_name(library)
        if safe:
            features[f"lib_{safe}"] = 1.0

    features["library_count"] = float(len(libraries))
    features["missing_library_count"] = float(len(missing_libraries))
    features["environment_risk"] = float(prompt_features.get("environment_risk", 0.0) or 0.0)
    features["plausibility_score"] = float(prompt_features.get("plausibility_score", 0.0) or 0.0)

    archive_signal = "archive" in token_set or any(
        library in {"gzip", "shutil", "tarfile", "zipfile"} for library in libraries
    )
    network_signal = "network" in categories or "http" in token_set or "url" in token_set
    plotting_signal = "plotting" in categories or "plot" in token_set
    filesystem_signal = "filesystem" in categories or bool({"file", "files"} & token_set)

    features["signal_archive"] = float(archive_signal)
    features["signal_network"] = float(network_signal)
    features["signal_plotting"] = float(plotting_signal)
    features["signal_filesystem"] = float(filesystem_signal)
    features["combo_network_archive"] = float(network_signal and archive_signal)
    features["combo_network_plotting"] = float(network_signal and plotting_signal)
    features["combo_network_filesystem"] = float(network_signal and filesystem_signal)
    return features


def feature_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    total = 0.0
    for key in keys:
        scale = 200.0 if key.startswith("length_") else 1.0
        delta = (left.get(key, 0.0) - right.get(key, 0.0)) / scale
        total += delta * delta
    return total
