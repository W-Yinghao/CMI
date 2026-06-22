"""Runner execution config + model spec. Every scientific field comes from the strict manifest; the
device and method execution order are runtime-only and never enter a hash. The model seed is bound by
the RunKey, so it is excluded from ``execution_config_hash``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..leakage.cache import critic_config_hash
from ..leakage.critic import CriticConfig
from ..train.engine import EngineConfig, engine_config_from_manifest
from ..train.rng import derive_seed
from .keys import canonical_json_hash

_METHODS = {"ERM", "OACI", "global_lpc", "uniform"}


@dataclass(frozen=True)
class ModelSpec:
    factory_id: str
    backbone_config: tuple                       # sorted (k, v) items
    input_shape: tuple
    n_classes: int
    model_spec_hash: str

    @staticmethod
    def build(factory_id, backbone_config, input_shape, n_classes) -> "ModelSpec":
        bc = tuple(sorted((str(k), v) for k, v in dict(backbone_config).items()))
        h = canonical_json_hash({"factory": str(factory_id), "backbone": bc,
                                 "input_shape": list(input_shape), "n_classes": int(n_classes)})
        return ModelSpec(str(factory_id), bc, tuple(int(s) for s in input_shape), int(n_classes), h)


@dataclass(frozen=True)
class RunnerExecutionConfig:
    engine_template: EngineConfig
    critic: CriticConfig
    method_critic_hidden: int
    global_lpc_alpha: float
    selection_score_tolerance: float
    feature_chunk_size: int | None
    prediction_chunk_size: int | None
    prediction_prob_floor: float
    ece_bins: int
    execution_config_hash: str

    def engine_config_for(self, run_key) -> EngineConfig:
        """Per-level engine config: the only thing the model seed changes is ``base_seed``."""
        return dataclasses.replace(self.engine_template,
                                   base_seed=derive_seed(run_key.model_seed, "engine", run_key.run_key_hash))

    @staticmethod
    def from_manifest(manifest) -> "RunnerExecutionConfig":
        names = list(manifest.methods.names or [])
        if len(names) != len(set(names)) or set(names) != _METHODS:
            raise ValueError(f"methods.names must be exactly {sorted(_METHODS)} with no duplicates; got {names}")
        eng = engine_config_from_manifest(manifest, base_seed=0)
        p = manifest.probe
        critic = CriticConfig(capacities=tuple(int(c) for c in p.capacities), l2_C=float(p.l2_C),
                              max_iter=int(p.max_iter), prob_floor=float(p.prob_floor),
                              feature_seed_base=int(p.feature_seed_base))
        h = canonical_json_hash({
            "engine": canonical_json_hash(dataclasses.asdict(dataclasses.replace(eng, base_seed=0))),
            "critic": critic_config_hash(critic),
            "method_critic_hidden": int(manifest.methods.critic_capacity),
            "global_lpc_alpha": float(manifest.methods.global_lpc_laplace_smoothing),
            "selection_score_tolerance": float(manifest.training.selection_score_tolerance),
            "feature_chunk_size": manifest.training.feature_chunk_size,
            "prediction_chunk_size": manifest.training.prediction_chunk_size,
            "prediction_prob_floor": float(manifest.evaluation.prediction_prob_floor),
            "ece_bins": int(manifest.evaluation.ece_bins)})
        return RunnerExecutionConfig(
            engine_template=eng, critic=critic, method_critic_hidden=int(manifest.methods.critic_capacity),
            global_lpc_alpha=float(manifest.methods.global_lpc_laplace_smoothing),
            selection_score_tolerance=float(manifest.training.selection_score_tolerance),
            feature_chunk_size=manifest.training.feature_chunk_size,
            prediction_chunk_size=manifest.training.prediction_chunk_size,
            prediction_prob_floor=float(manifest.evaluation.prediction_prob_floor),
            ece_bins=int(manifest.evaluation.ece_bins), execution_config_hash=h)
