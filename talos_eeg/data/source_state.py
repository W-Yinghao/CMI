"""Source-state schema for TALOS frozen-feature replay.

TALOS_00A may construct this state from synthetic or frozen source rows for P0
replay. A source-free deployment claim remains false until a future serialized
source-state guard is approved and passed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cedar_eeg.data.feature_schema import stable_json_hash


SOURCE_STATE_MODE_P0_REPLAY = "constructed_from_frozen_source_features_for_P0_replay"


@dataclass(frozen=True)
class SourceState:
    n_features: int
    n_classes: int
    class_labels: tuple[int, ...]
    class_prototypes: np.ndarray
    class_diag_var: np.ndarray
    source_prior: np.ndarray
    source_prior_low: np.ndarray
    source_prior_high: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    source_state_mode: str = SOURCE_STATE_MODE_P0_REPLAY
    source_free_deployment_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": int(self.n_features),
            "n_classes": int(self.n_classes),
            "class_labels": [int(x) for x in self.class_labels],
            "class_prototypes": self.class_prototypes.astype(float).tolist(),
            "class_diag_var": self.class_diag_var.astype(float).tolist(),
            "source_prior": self.source_prior.astype(float).tolist(),
            "source_prior_low": self.source_prior_low.astype(float).tolist(),
            "source_prior_high": self.source_prior_high.astype(float).tolist(),
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_std": self.feature_std.astype(float).tolist(),
            "readout_weight": self.readout_weight.astype(float).tolist(),
            "readout_bias": self.readout_bias.astype(float).tolist(),
            "source_state_mode": self.source_state_mode,
            "source_free_deployment_claim": bool(self.source_free_deployment_claim),
        }

    def hash(self) -> str:
        return stable_json_hash(self.to_dict())


@dataclass(frozen=True)
class SourceStateSchemaResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    source_state_hash: str
    source_state_mode: str
    source_free_deployment_claim: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_2d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_labels(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).astype(np.int64, copy=False)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("y_source must be a non-empty 1D array")
    if np.any(arr < 0):
        raise ValueError("y_source labels must be non-negative integers")
    return arr


def source_logits(state: SourceState, z: np.ndarray) -> np.ndarray:
    z = _as_2d_float("z", z)
    if z.shape[1] != state.n_features:
        raise ValueError(f"z has {z.shape[1]} features, expected {state.n_features}")
    return z @ state.readout_weight + state.readout_bias


def build_source_state(
    z_source: np.ndarray,
    y_source: np.ndarray,
    *,
    min_var: float = 1e-4,
    prior_interval_radius: float = 0.15,
) -> SourceState:
    z = _as_2d_float("z_source", z_source)
    y = _as_labels(y_source)
    if len(y) != len(z):
        raise ValueError("z_source and y_source length mismatch")
    labels = tuple(int(x) for x in sorted(np.unique(y)))
    if len(labels) < 2:
        raise ValueError("at least two source classes are required")

    prototypes = []
    diag_var = []
    counts = []
    for label in labels:
        rows = z[y == label]
        if len(rows) == 0:
            raise ValueError(f"missing rows for class {label}")
        prototypes.append(rows.mean(axis=0))
        diag_var.append(np.maximum(rows.var(axis=0), min_var))
        counts.append(len(rows))

    prototypes_arr = np.vstack(prototypes)
    diag_var_arr = np.vstack(diag_var)
    counts_arr = np.asarray(counts, dtype=np.float64)
    prior = counts_arr / counts_arr.sum()
    prior_low = np.clip(prior - prior_interval_radius, 0.0, 1.0)
    prior_high = np.clip(prior + prior_interval_radius, 0.0, 1.0)
    feature_mean = z.mean(axis=0)
    feature_std = np.maximum(z.std(axis=0), np.sqrt(min_var))

    pooled_var = np.maximum(z.var(axis=0), min_var)
    weight = (prototypes_arr / pooled_var).T
    bias = -0.5 * np.sum((prototypes_arr * prototypes_arr) / pooled_var, axis=1)
    bias = bias + np.log(np.maximum(prior, 1e-8))

    state = SourceState(
        n_features=int(z.shape[1]),
        n_classes=int(len(labels)),
        class_labels=labels,
        class_prototypes=prototypes_arr,
        class_diag_var=diag_var_arr,
        source_prior=prior,
        source_prior_low=prior_low,
        source_prior_high=prior_high,
        feature_mean=feature_mean,
        feature_std=feature_std,
        readout_weight=weight,
        readout_bias=bias,
    )
    validate_source_state_schema(state)
    return state


def validate_source_state_schema(state: SourceState) -> SourceStateSchemaResult:
    checks: list[str] = []
    warnings: list[str] = []
    if state.source_state_mode != SOURCE_STATE_MODE_P0_REPLAY:
        raise ValueError(f"unexpected source_state_mode: {state.source_state_mode}")
    checks.append("source_state_mode_p0_replay")
    if state.source_free_deployment_claim is not False:
        raise ValueError("TALOS_00A source state must not claim source-free deployment")
    checks.append("source_free_deployment_claim_false")

    expected_vectors = {
        "source_prior": state.source_prior,
        "source_prior_low": state.source_prior_low,
        "source_prior_high": state.source_prior_high,
        "feature_mean": state.feature_mean,
        "feature_std": state.feature_std,
        "readout_bias": state.readout_bias,
    }
    for name, arr in expected_vectors.items():
        arr = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    checks.append("finite_vector_fields")

    if state.class_prototypes.shape != (state.n_classes, state.n_features):
        raise ValueError("class_prototypes shape mismatch")
    if state.class_diag_var.shape != (state.n_classes, state.n_features):
        raise ValueError("class_diag_var shape mismatch")
    if state.readout_weight.shape != (state.n_features, state.n_classes):
        raise ValueError("readout_weight shape mismatch")
    if state.readout_bias.shape != (state.n_classes,):
        raise ValueError("readout_bias shape mismatch")
    checks.append("array_shapes")

    if not np.isclose(float(state.source_prior.sum()), 1.0):
        raise ValueError("source_prior must sum to 1")
    if np.any(state.source_prior < 0.0):
        raise ValueError("source_prior must be non-negative")
    checks.append("source_prior_simplex")

    if np.any(state.class_diag_var <= 0.0) or np.any(state.feature_std <= 0.0):
        raise ValueError("variance/std fields must be positive")
    checks.append("positive_scale_fields")

    return SourceStateSchemaResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        source_state_hash=state.hash(),
        source_state_mode=state.source_state_mode,
        source_free_deployment_claim=state.source_free_deployment_claim,
    )
