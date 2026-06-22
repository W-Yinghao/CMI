"""Confirmatory protocol manifest. EVERY field below must be filled explicitly (no paper-level
defaults in code) — ``validate_complete`` raises if any is missing. ``freeze`` then emits canonical
JSON + a full SHA-256.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

REQUIRED = [
    "protocol_id", "datasets", "outer_target_factor", "domain_factor", "group_factor",
    "support_unit", "eval_unit", "channel_list", "preprocessing", "support_m", "risk_metric",
    "epsilon", "delta_bacc", "alpha", "n_bootstrap", "split_seed", "deletion_seed",
    "model_seeds", "methods", "backbone", "training_budget", "k1_rule", "k2_rule",
]


@dataclass
class ProtocolManifest:
    protocol_id: str | None = None
    datasets: list | None = None
    outer_target_factor: str | None = None
    domain_factor: str | None = None
    group_factor: str | None = None
    support_unit: str | None = None
    eval_unit: str | None = None
    channel_list: list | None = None
    preprocessing: dict | None = None
    support_m: int | None = None
    risk_metric: str | None = None
    epsilon: float | None = None
    delta_bacc: float | None = None
    alpha: float | None = None
    n_bootstrap: int | None = None
    split_seed: int | None = None
    deletion_seed: int | None = None
    model_seeds: list | None = None
    methods: list | None = None
    backbone: str | None = None
    training_budget: dict | None = None
    k1_rule: str | None = None
    k2_rule: str | None = None

    def missing_fields(self) -> list:
        return [f for f in REQUIRED if getattr(self, f) in (None, [], {}, "")]

    def validate_complete(self) -> "ProtocolManifest":
        miss = self.missing_fields()
        if miss:
            raise ValueError(f"confirmatory manifest is incomplete; fill explicitly: {miss}")
        if self.split_seed == self.deletion_seed:
            raise ValueError("split_seed and deletion_seed must differ (separate randomness blocks)")
        return self

    def to_canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)
