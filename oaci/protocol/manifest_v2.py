"""Protocol v2 — heterogeneous, machine-expressible: one fully-specified block PER dataset
(``support_m``, epoch/window, class map, channel list, target/domain/group/unit factors are all
dataset-specific). The runner can validate and execute each block; the v1 "clinical override is a
comment" problem is gone. A ``status: smoke`` manifest is REJECTED in confirmatory mode (the smoke
manifest must be a separate file, not a temporarily-shortened confirmatory one).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

DATASET_REQUIRED = ["enabled", "cohort_ids", "class_names", "outer_target_factor", "domain_factor",
                    "group_factor", "support_unit_factor", "eval_unit_factor", "support_m",
                    "channels", "preprocessing"]
TOP_REQUIRED = ["protocol_id", "status", "datasets", "risk", "methods", "evaluation", "k1", "k2"]


@dataclass
class DatasetBlock:
    enabled: bool | None = None
    cohort_ids: list | None = None
    class_names: list | None = None
    outer_target_factor: str | None = None
    domain_factor: str | None = None
    group_factor: str | None = None
    support_unit_factor: str | None = None
    eval_unit_factor: str | None = None
    support_m: int | None = None
    channels: list | None = None
    preprocessing: dict | None = None

    def missing(self) -> list:
        return [f for f in DATASET_REQUIRED if getattr(self, f) in (None, [], {}, "")]


@dataclass
class ProtocolManifestV2:
    protocol_id: str | None = None
    status: str | None = None
    datasets: dict | None = None          # name -> DatasetBlock
    risk: dict | None = None
    methods: dict | None = None
    evaluation: dict | None = None
    k1: dict | None = None
    k2: dict | None = None

    def enabled_datasets(self) -> dict:
        return {k: v for k, v in (self.datasets or {}).items() if getattr(v, "enabled", False)}

    def validate_complete(self) -> "ProtocolManifestV2":
        top_miss = [f for f in TOP_REQUIRED if getattr(self, f) in (None, {}, "")]
        if top_miss:
            raise ValueError(f"protocol v2 missing top-level fields: {top_miss}")
        en = self.enabled_datasets()
        if not en:
            raise ValueError("protocol v2 has no enabled datasets")
        for name, blk in en.items():
            miss = blk.missing()
            if miss:
                raise ValueError(f"dataset block {name!r} incomplete: {miss}")
        return self

    def assert_confirmatory(self) -> "ProtocolManifestV2":
        if self.status == "smoke":
            raise ValueError("a status='smoke' manifest cannot be used in confirmatory mode "
                             "(use a separate smoke manifest, not a shortened confirmatory one)")
        return self.validate_complete()

    def to_canonical_json(self) -> str:
        d = asdict(self)
        d["datasets"] = {k: asdict(v) for k, v in (self.datasets or {}).items()}
        return json.dumps(d, sort_keys=True, default=str)

    def freeze(self) -> dict:
        self.validate_complete()
        canon = self.to_canonical_json()
        return {"canonical_json": canon, "sha256": hashlib.sha256(canon.encode()).hexdigest()}


def load_v2(path: str) -> ProtocolManifestV2:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PyYAML required to load {path}: {e}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    ds = {name: DatasetBlock(**blk) for name, blk in (d.get("datasets") or {}).items()}
    return ProtocolManifestV2(
        protocol_id=d.get("protocol_id"), status=d.get("status"), datasets=ds,
        risk=d.get("risk"), methods=d.get("methods"), evaluation=d.get("evaluation"),
        k1=d.get("k1"), k2=d.get("k2"),
    )
