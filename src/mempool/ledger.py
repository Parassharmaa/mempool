from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LedgerEvent:
    type: str
    payload: dict[str, Any]
    task_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class JsonlLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: LedgerEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
