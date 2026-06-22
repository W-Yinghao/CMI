"""Run / fold keys + canonical identity helpers (one place, SHA-256, never Python ``hash()``).

FoldKey deliberately EXCLUDES the model seed: the data split, deletion schedule, frozen maps,
source-audit plans and target signatures are identical across model seeds. The model seed lives on
RunKey and only affects training plans / initialisation / checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


def feed_string(h, value) -> None:
    b = str(value).encode("utf-8"); h.update(len(b).to_bytes(8, "little")); h.update(b)


def feed_int64(h, value) -> None:
    h.update(int(value).to_bytes(8, "little", signed=True))


def feed_float64(h, value) -> None:
    import struct
    h.update(struct.pack("<d", float(value)))


def canonical_json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class FoldKey:
    manifest_hash: str
    dataset_id: str
    outer_fold: str
    split_seed: int
    deletion_seed: int

    @property
    def fold_key_hash(self) -> str:
        return canonical_json_hash({"fold_key": asdict(self)})


@dataclass(frozen=True)
class RunKey:
    fold_key: FoldKey
    deletion_level: int
    model_seed: int

    @property
    def run_key_hash(self) -> str:
        return canonical_json_hash({"fold_key": asdict(self.fold_key),
                                    "deletion_level": int(self.deletion_level),
                                    "model_seed": int(self.model_seed)})
