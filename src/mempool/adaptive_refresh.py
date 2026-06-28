from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REQUIRED_GUARDRAILS = {
    "immutable_raw_traces",
    "versioned_distilled_dataset",
    "evaluation_before_promotion",
    "rollback_point",
    "privacy_filter",
    "separate_user_memory",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exists(path_value: str | None) -> bool:
    return bool(path_value) and Path(path_value).exists()


def _guardrail_checks(
    *,
    ledger_path: Path | None,
    distilled_dataset_path: Path,
    candidate_model_path: Path,
    gate_path: Path,
    active_registry_path: Path,
    privacy_manifest_path: Path | None,
) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "immutable_raw_traces",
            "passed": ledger_path is not None and ledger_path.exists(),
            "evidence": str(ledger_path) if ledger_path else None,
            "reason": None if ledger_path is not None and ledger_path.exists() else "missing raw ledger path",
        },
        {
            "name": "versioned_distilled_dataset",
            "passed": distilled_dataset_path.exists(),
            "evidence": str(distilled_dataset_path),
            "reason": None if distilled_dataset_path.exists() else "missing distilled dataset",
        },
        {
            "name": "evaluation_before_promotion",
            "passed": gate_path.exists(),
            "evidence": str(gate_path),
            "reason": None if gate_path.exists() else "missing policy gate report",
        },
        {
            "name": "rollback_point",
            "passed": active_registry_path.exists(),
            "evidence": str(active_registry_path),
            "reason": None if active_registry_path.exists() else "missing active policy registry",
        },
        {
            "name": "privacy_filter",
            "passed": privacy_manifest_path is not None and privacy_manifest_path.exists(),
            "evidence": str(privacy_manifest_path) if privacy_manifest_path else None,
            "reason": None if privacy_manifest_path is not None and privacy_manifest_path.exists() else "missing privacy manifest",
        },
        {
            "name": "separate_user_memory",
            "passed": privacy_manifest_path is not None and privacy_manifest_path.exists(),
            "evidence": str(privacy_manifest_path) if privacy_manifest_path else None,
            "reason": None if privacy_manifest_path is not None and privacy_manifest_path.exists() else "missing memory-scope declaration",
        },
        {
            "name": "candidate_artifact",
            "passed": candidate_model_path.exists(),
            "evidence": str(candidate_model_path),
            "reason": None if candidate_model_path.exists() else "missing candidate model",
        },
    ]
    if privacy_manifest_path is not None and privacy_manifest_path.exists():
        privacy = read_json(privacy_manifest_path)
        scope = str(privacy.get("memory_scope", ""))
        contains_raw_private_text = bool(privacy.get("contains_raw_private_text", True))
        explicit_approval = bool(privacy.get("explicit_private_training_approval", False))
        for check in checks:
            if check["name"] == "privacy_filter":
                check["passed"] = not contains_raw_private_text or explicit_approval
                check["reason"] = None if check["passed"] else "raw private text lacks explicit training approval"
                check["evidence"] = {
                    "manifest": str(privacy_manifest_path),
                    "contains_raw_private_text": contains_raw_private_text,
                    "explicit_private_training_approval": explicit_approval,
                    "redaction_policy": privacy.get("redaction_policy"),
                }
            elif check["name"] == "separate_user_memory":
                check["passed"] = scope in {"benchmark", "general", "user_specific"}
                check["reason"] = None if check["passed"] else f"unknown memory scope: {scope}"
                check["evidence"] = {
                    "manifest": str(privacy_manifest_path),
                    "memory_scope": scope,
                    "training_scope": privacy.get("training_scope"),
                }
    return checks


def build_privacy_manifest(
    *,
    distilled_dataset_path: Path,
    output_path: Path,
    memory_scope: str = "benchmark",
    training_scope: str = "general",
    contains_raw_private_text: bool = False,
    explicit_private_training_approval: bool = False,
    redaction_policy: str = "benchmark prompts and aggregate worker metadata only",
) -> dict[str, Any]:
    manifest = {
        "schema_version": "mempool.privacy_manifest.v1",
        "distilled_dataset": str(distilled_dataset_path),
        "memory_scope": memory_scope,
        "training_scope": training_scope,
        "contains_raw_private_text": contains_raw_private_text,
        "explicit_private_training_approval": explicit_private_training_approval,
        "redaction_policy": redaction_policy,
        "created_at": datetime.now(UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_refresh_cycle(
    *,
    cycle_id: str,
    distilled_dataset_path: Path,
    candidate_model_path: Path,
    candidate_report_path: Path,
    gate_path: Path,
    active_registry_path: Path,
    output_path: Path,
    ledger_path: Path | None = None,
    privacy_manifest_path: Path | None = None,
) -> dict[str, Any]:
    gate = read_json(gate_path)
    report = read_json(candidate_report_path)
    guardrails = _guardrail_checks(
        ledger_path=ledger_path,
        distilled_dataset_path=distilled_dataset_path,
        candidate_model_path=candidate_model_path,
        gate_path=gate_path,
        active_registry_path=active_registry_path,
        privacy_manifest_path=privacy_manifest_path,
    )
    failed_guardrails = [
        check["name"]
        for check in guardrails
        if check["name"] in REQUIRED_GUARDRAILS and not check["passed"]
    ]
    gate_decision = str(gate.get("decision"))
    promotion_profile = gate.get("promotion_profile")
    decision = "promote" if gate_decision == "promote" and not failed_guardrails else "quarantine"
    reasons = list(gate.get("reasons") or [])
    for check in guardrails:
        if check["name"] in REQUIRED_GUARDRAILS and not check["passed"]:
            reasons.append(f"guardrail {check['name']}: {check['reason']}")
    cycle = {
        "schema_version": "mempool.adaptive_refresh_cycle.v1",
        "cycle_id": cycle_id,
        "created_at": datetime.now(UTC).isoformat(),
        "decision": decision,
        "gate_decision": gate_decision,
        "promotion_profile": promotion_profile,
        "reasons": reasons,
        "artifacts": {
            "ledger": str(ledger_path) if ledger_path else None,
            "distilled_dataset": str(distilled_dataset_path),
            "candidate_model": str(candidate_model_path),
            "candidate_report": str(candidate_report_path),
            "policy_gate": str(gate_path),
            "active_registry": str(active_registry_path),
            "privacy_manifest": str(privacy_manifest_path) if privacy_manifest_path else None,
        },
        "candidate_metrics": {
            "evaluation": report.get("evaluation"),
            "leave_one_out": report.get("leave_one_out"),
        },
        "gate": gate,
        "guardrails": guardrails,
        "rollback": {
            "available": active_registry_path.exists(),
            "command": f"PYTHONPATH=src python3 tools/policy_registry.py --registry {active_registry_path} rollback",
        },
        "promotion": {
            "allowed": decision == "promote",
            "profile": promotion_profile,
            "command": f"PYTHONPATH=src python3 tools/policy_registry.py --registry {active_registry_path} apply-refresh --refresh {gate_path}",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cycle
