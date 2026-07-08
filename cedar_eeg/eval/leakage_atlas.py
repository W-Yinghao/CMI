"""Leakage atlas serialization for CEDAR audits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AtlasRecord:
    unit_type: str
    unit_id: str
    domain_score: float
    task_score: float
    cedar_score: float
    selected_for_drop: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def latent_scores_to_atlas(scores: list[dict[str, float | int]], dropped: set[int] | None = None) -> list[AtlasRecord]:
    dropped = dropped or set()
    records = []
    for rec in scores:
        dim = int(rec["dim"])
        records.append(
            AtlasRecord(
                unit_type="latent_dim",
                unit_id=str(dim),
                domain_score=float(rec["domain_score"]),
                task_score=float(rec["task_score"]),
                cedar_score=float(rec["cedar_score"]),
                selected_for_drop=dim in dropped,
            )
        )
    return records


def write_atlas_json(path: str | Path, records: list[AtlasRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, sort_keys=True)
