from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .multi_head_orchestrator import read_substrate, validate_substrate_records


SCHEMA_VERSION = "mempool.qwen_logits_orchestrator_plan.v1"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class QwenLogitsTrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    backend: str = "transformers"
    train_backbone: bool = False
    lora_rank: int = 0
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    max_length: int = 1536
    seed: int = 7
    device: str = "auto"


def optional_dependency_status() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ["torch", "transformers", "mlx", "mlx_lm"]
    }


def training_dependencies_available(backend: str) -> bool:
    status = optional_dependency_status()
    if backend == "transformers":
        return bool(status["torch"] and status["transformers"])
    if backend == "mlx":
        return bool(status["mlx"] and status["mlx_lm"])
    raise ValueError(f"unsupported backend: {backend}")


def audit_qwen_training_readiness(
    *,
    backend: str = "transformers",
    require_gpu: bool = False,
) -> dict[str, Any]:
    if backend not in {"transformers", "mlx"}:
        raise ValueError(f"unsupported backend: {backend}")
    dependency_status = optional_dependency_status()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_supported = sys.version_info < (3, 13)
    backend_ready = training_dependencies_available(backend)
    reasons = []
    recommendations = []

    if backend == "transformers":
        missing = [name for name in ["torch", "transformers"] if not dependency_status[name]]
        if missing:
            reasons.append(f"missing transformers backend dependencies: {', '.join(missing)}")
            recommendations.append("create a Python 3.11 or 3.12 environment and install `.[qwen-train]`")
        if not python_supported:
            reasons.append(f"current Python {python_version} may not have stable PyTorch wheels")
            recommendations.append("prefer Python 3.11 or 3.12 for the first Qwen-small training run")
    else:
        missing = [name for name in ["mlx", "mlx_lm"] if not dependency_status[name]]
        if missing:
            reasons.append(f"missing MLX backend dependencies: {', '.join(missing)}")
            recommendations.append("install `.[mlx-train]` on Apple Silicon or use a GPU host")

    if require_gpu:
        reasons.append("GPU/accelerator availability was requested but is not verified by this lightweight audit")
        recommendations.append("run the training job on a GPU or Apple MLX machine for practical turnaround")

    if not reasons and not require_gpu:
        recommendations.append("run a frozen-backbone head-training smoke before enabling LoRA or backbone updates")

    return {
        "schema_version": "mempool.qwen_training_readiness.v1",
        "backend": backend,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": python_version,
        "python_supported_for_torch": python_supported,
        "dependency_status": dependency_status,
        "backend_ready": backend_ready,
        "require_gpu": require_gpu,
        "ready_for_local_head_training": backend_ready and python_supported and not require_gpu,
        "reasons": reasons,
        "recommendations": recommendations,
    }


def decision_text(example: dict[str, Any]) -> str:
    messages = example.get("messages") or []
    user_parts = [str(message.get("content") or "") for message in messages if message.get("role") == "user"]
    if user_parts:
        return "\n\n".join(user_parts)
    return str(example.get("prompt") or example.get("task_id") or "")


def target_summary(example: dict[str, Any]) -> dict[str, Any]:
    target = example.get("target") or {}
    return {
        "worker_distribution": target.get("worker_distribution") or {},
        "workflow_distribution": target.get("workflow_distribution") or {},
        "verifier_probability": float(target.get("verifier_probability", 0.0) or 0.0),
        "abstain_probability": float(target.get("abstain_probability", 0.0) or 0.0),
    }


def build_training_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in records:
        rows.append(
            {
                "schema_version": "mempool.qwen_logits_orchestrator_row.v1",
                "task_id": example.get("task_id"),
                "text": decision_text(example),
                "target": target_summary(example),
            }
        )
    return rows


def align_hidden_dtype(hidden: Any, reference_weight: Any) -> Any:
    return hidden.to(dtype=reference_weight.dtype)


def resolve_torch_device(torch_module: Any, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_qwen_logits_training_plan(
    *,
    substrate_path: Path,
    output_path: Path,
    rows_output_path: Path,
    config: QwenLogitsTrainingConfig,
) -> dict[str, Any]:
    records = read_substrate(substrate_path)
    errors = validate_substrate_records(records)
    if errors:
        raise ValueError(f"invalid substrate: {errors}")
    rows = build_training_rows(records)
    worker_ids = sorted({worker_id for row in rows for worker_id in row["target"]["worker_distribution"]})
    workflow_labels = sorted({label for row in rows for label in row["target"]["workflow_distribution"]})
    dependency_status = optional_dependency_status()
    can_train_here = training_dependencies_available(config.backend)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "substrate": str(substrate_path),
        "rows_output": str(rows_output_path),
        "record_count": len(rows),
        "worker_ids": worker_ids,
        "workflow_labels": workflow_labels,
        "config": asdict(config),
        "dependency_status": dependency_status,
        "can_train_here": can_train_here,
        "training_order": [
            "freeze Qwen-small backbone",
            "train worker/workflow/verifier/abstain heads on measured soft targets",
            "compare held-out routing against the linear multi-head baseline",
            "only enable LoRA/backbone updates after the heads beat the baseline",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def build_qwen_logits_training_plan_from_rows(
    *,
    rows_path: Path,
    output_path: Path,
    config: QwenLogitsTrainingConfig,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    worker_ids = sorted({worker_id for row in rows for worker_id in row["target"]["worker_distribution"]})
    workflow_labels = sorted({label for row in rows for label in row["target"]["workflow_distribution"]})
    dependency_status = optional_dependency_status()
    plan = {
        "schema_version": SCHEMA_VERSION,
        "prepared_rows": str(rows_path),
        "rows_output": str(rows_path),
        "record_count": len(rows),
        "worker_ids": worker_ids,
        "workflow_labels": workflow_labels,
        "config": asdict(config),
        "dependency_status": dependency_status,
        "can_train_here": training_dependencies_available(config.backend),
        "training_order": [
            "freeze Qwen-small backbone",
            "train worker/workflow/verifier/abstain heads on measured soft targets",
            "compare held-out routing against the linear multi-head baseline",
            "only enable LoRA/backbone updates after the heads beat the baseline",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def run_transformers_training(
    *,
    training_rows_path: Path,
    output_dir: Path,
    config: QwenLogitsTrainingConfig,
) -> dict[str, Any]:
    if not training_dependencies_available("transformers"):
        raise RuntimeError("transformers training requires installed torch and transformers packages")

    # Heavy ML imports stay inside the training path so normal repo tests remain
    # dependency-free.
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

    device = resolve_torch_device(torch, config.device)
    rows = [
        json.loads(line)
        for line in training_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    worker_ids = sorted({worker_id for row in rows for worker_id in row["target"]["worker_distribution"]})
    workflow_labels = sorted({label for row in rows for label in row["target"]["workflow_distribution"]})

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    backbone = AutoModel.from_pretrained(config.base_model)
    if not config.train_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    hidden_size = int(backbone.config.hidden_size)

    class HeadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.worker_head = nn.Linear(hidden_size, len(worker_ids))
            self.workflow_head = nn.Linear(hidden_size, len(workflow_labels))
            self.verifier_head = nn.Linear(hidden_size, 1)
            self.abstain_head = nn.Linear(hidden_size, 1)

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            hidden = outputs.last_hidden_state[torch.arange(input_ids.shape[0]), lengths]
            hidden = align_hidden_dtype(hidden, self.worker_head.weight)
            return {
                "worker_logits": self.worker_head(hidden),
                "workflow_logits": self.workflow_head(hidden),
                "verifier_logits": self.verifier_head(hidden).squeeze(-1),
                "abstain_logits": self.abstain_head(hidden).squeeze(-1),
            }

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=config.max_length,
            padding=True,
            return_tensors="pt",
        )
        worker_targets = torch.tensor(
            [
                [float(row["target"]["worker_distribution"].get(worker_id, 0.0)) for worker_id in worker_ids]
                for row in batch
            ],
            dtype=torch.float32,
        )
        workflow_targets = torch.tensor(
            [
                [float(row["target"]["workflow_distribution"].get(label, 0.0)) for label in workflow_labels]
                for row in batch
            ],
            dtype=torch.float32,
        )
        verifier_targets = torch.tensor(
            [float(row["target"]["verifier_probability"]) for row in batch],
            dtype=torch.float32,
        )
        abstain_targets = torch.tensor(
            [float(row["target"]["abstain_probability"]) for row in batch],
            dtype=torch.float32,
        )
        return {
            **encoded,
            "worker_targets": worker_targets,
            "workflow_targets": workflow_targets,
            "verifier_targets": verifier_targets,
            "abstain_targets": abstain_targets,
        }

    torch.manual_seed(config.seed)
    model = HeadModel().to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    loader = DataLoader(rows, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    history = []
    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            outputs = model(batch["input_ids"], batch["attention_mask"])
            worker_log_probs = torch.log_softmax(outputs["worker_logits"], dim=-1)
            workflow_log_probs = torch.log_softmax(outputs["workflow_logits"], dim=-1)
            loss = -(batch["worker_targets"] * worker_log_probs).sum(dim=-1).mean()
            loss = loss - (batch["workflow_targets"] * workflow_log_probs).sum(dim=-1).mean()
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                outputs["verifier_logits"],
                batch["verifier_targets"],
            )
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                outputs["abstain_logits"],
                batch["abstain_targets"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": total_loss / max(1, len(loader))})

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "mempool.qwen_logits_orchestrator_checkpoint.v1",
            "base_model": config.base_model,
            "worker_ids": worker_ids,
            "workflow_labels": workflow_labels,
            "head_state_dict": {
                key: value.cpu()
                for key, value in model.state_dict().items()
                if not key.startswith("backbone.")
            },
            "config": asdict(config),
            "history": history,
        },
        output_dir / "qwen_logits_heads.pt",
    )
    report = {
        "schema_version": "mempool.qwen_logits_orchestrator_train_report.v1",
        "output_dir": str(output_dir),
        "record_count": len(rows),
        "worker_ids": worker_ids,
        "workflow_labels": workflow_labels,
        "device": device,
        "history": history,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_transformers_evaluation(
    *,
    training_rows_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not training_dependencies_available("transformers"):
        raise RuntimeError("transformers evaluation requires installed torch and transformers packages")

    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = QwenLogitsTrainingConfig(**checkpoint["config"])
    device = resolve_torch_device(torch, config.device)
    rows = [
        json.loads(line)
        for line in training_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    worker_ids = list(checkpoint["worker_ids"])
    workflow_labels = list(checkpoint["workflow_labels"])

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    backbone = AutoModel.from_pretrained(config.base_model)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    hidden_size = int(backbone.config.hidden_size)

    class HeadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.worker_head = nn.Linear(hidden_size, len(worker_ids))
            self.workflow_head = nn.Linear(hidden_size, len(workflow_labels))
            self.verifier_head = nn.Linear(hidden_size, 1)
            self.abstain_head = nn.Linear(hidden_size, 1)

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            hidden = outputs.last_hidden_state[torch.arange(input_ids.shape[0]), lengths]
            hidden = align_hidden_dtype(hidden, self.worker_head.weight)
            return {
                "worker_logits": self.worker_head(hidden),
                "workflow_logits": self.workflow_head(hidden),
                "verifier_logits": self.verifier_head(hidden).squeeze(-1),
                "abstain_logits": self.abstain_head(hidden).squeeze(-1),
            }

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=config.max_length,
            padding=True,
            return_tensors="pt",
        )
        worker_targets = torch.tensor(
            [
                [float(row["target"]["worker_distribution"].get(worker_id, 0.0)) for worker_id in worker_ids]
                for row in batch
            ],
            dtype=torch.float32,
        )
        workflow_targets = torch.tensor(
            [
                [float(row["target"]["workflow_distribution"].get(label, 0.0)) for label in workflow_labels]
                for row in batch
            ],
            dtype=torch.float32,
        )
        return {
            **encoded,
            "worker_targets": worker_targets,
            "workflow_targets": workflow_targets,
        }

    model = HeadModel().to(device)
    model.load_state_dict(checkpoint["head_state_dict"], strict=False)
    model.eval()
    loader = DataLoader(rows, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    worker_correct = 0
    workflow_correct = 0
    total = 0
    worker_loss_total = 0.0
    workflow_loss_total = 0.0
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            outputs = model(batch["input_ids"], batch["attention_mask"])
            worker_probs = torch.softmax(outputs["worker_logits"], dim=-1)
            workflow_probs = torch.softmax(outputs["workflow_logits"], dim=-1)
            worker_targets = batch["worker_targets"]
            workflow_targets = batch["workflow_targets"]
            worker_loss = -(worker_targets * torch.log(worker_probs.clamp_min(1e-12))).sum(dim=-1)
            workflow_loss = -(workflow_targets * torch.log(workflow_probs.clamp_min(1e-12))).sum(dim=-1)
            worker_predicted = worker_probs.argmax(dim=-1)
            worker_target = worker_targets.argmax(dim=-1)
            workflow_predicted = workflow_probs.argmax(dim=-1)
            workflow_target = workflow_targets.argmax(dim=-1)
            worker_correct += int((worker_predicted == worker_target).sum().item())
            workflow_correct += int((workflow_predicted == workflow_target).sum().item())
            worker_loss_total += float(worker_loss.sum().item())
            workflow_loss_total += float(workflow_loss.sum().item())
            for index in range(worker_probs.shape[0]):
                predictions.append(
                    {
                        "predicted_worker_id": worker_ids[int(worker_predicted[index].item())],
                        "target_worker_id": worker_ids[int(worker_target[index].item())],
                        "predicted_workflow": workflow_labels[int(workflow_predicted[index].item())],
                        "target_workflow": workflow_labels[int(workflow_target[index].item())],
                    }
                )
            total += int(worker_probs.shape[0])

    report = {
        "schema_version": "mempool.qwen_logits_orchestrator_eval_report.v1",
        "checkpoint": str(checkpoint_path),
        "rows": str(training_rows_path),
        "record_count": total,
        "device": device,
        "worker_accuracy": worker_correct / total if total else 0.0,
        "workflow_accuracy": workflow_correct / total if total else 0.0,
        "mean_worker_loss": worker_loss_total / total if total else 0.0,
        "mean_workflow_loss": workflow_loss_total / total if total else 0.0,
        "worker_ids": worker_ids,
        "workflow_labels": workflow_labels,
        "prediction_sample": predictions[:10],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
