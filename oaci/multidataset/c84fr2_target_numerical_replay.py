"""Numerical and persistence contracts for the C84FR2 target-only stage."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ARTIFACT_SCHEMA = "c84f_target_unlabeled_v2"
CONTEXT_DIGEST_SCHEMA = "c84f_target_context_and_digest_index_v2"
SAME_BACKEND_TOLERANCE = 1e-6
STRICT_REPLAY_TOLERANCE = 1e-6

TARGET_ARRAY_FIELDS = (
    "unit_id", "dataset", "panel", "training_seed", "level",
    "level_intervention_id", "regime", "epoch", "trajectory_order",
    "target_subject_id", "target_trial_id", "session", "run",
    "logits", "probabilities", "z", "Wz_plus_b", "classifier_weight",
    "classifier_bias", "repeat_logits", "repeat_z",
)


class C84FR2NumericalError(RuntimeError):
    """Raised when same-backend identity or exact persistence fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C84FR2NumericalError(message)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _require_exact_fields(arrays: Mapping[str, Any]) -> None:
    observed = set(arrays)
    expected = set(TARGET_ARRAY_FIELDS)
    _require(
        observed == expected,
        f"target artifact field-set drift: missing={sorted(expected-observed)} "
        f"unknown={sorted(observed-expected)}",
    )


def canonical_array_descriptor(value: Any, *, np: Any) -> dict[str, Any]:
    """Return an exact dtype/shape/content identity for one persisted field."""
    array = np.asarray(value)
    _require(not array.dtype.hasobject, "object dtype is forbidden in target artifacts")
    contiguous = np.ascontiguousarray(array)
    if np.issubdtype(contiguous.dtype, np.number):
        _require(bool(np.isfinite(contiguous).all()), "nonfinite target artifact array")
    header = {
        "dtype": contiguous.dtype.str,
        "shape": [int(value) for value in contiguous.shape],
    }
    payload = _canonical_json(header) + b"\0" + contiguous.tobytes(order="C")
    return {
        **header,
        "bytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_digest_registry(arrays: Mapping[str, Any], *, np: Any) -> dict[str, dict[str, Any]]:
    _require_exact_fields(arrays)
    return {
        field: canonical_array_descriptor(arrays[field], np=np)
        for field in TARGET_ARRAY_FIELDS
    }


def verify_digest_registry(
    expected: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
    *,
    np: Any,
) -> dict[str, dict[str, Any]]:
    _require(set(expected) == set(TARGET_ARRAY_FIELDS), "pre-write digest registry field-set drift")
    observed = build_digest_registry(arrays, np=np)
    mismatch = {
        field: {"expected": dict(expected[field]), "observed": observed[field]}
        for field in TARGET_ARRAY_FIELDS
        if dict(expected[field]) != observed[field]
    }
    _require(not mismatch, f"persisted target array digest mismatch: {sorted(mismatch)}")
    return observed


def _finite_max_abs(left: Any, right: Any, *, np: Any, name: str) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    _require(a.shape == b.shape, f"{name} shape drift: {a.shape} != {b.shape}")
    _require(bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()), f"{name} is nonfinite")
    if not a.size:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def validate_same_backend_tensors(
    logits_model: Any,
    logits_direct: Any,
    *,
    torch: Any,
    require_cuda: bool,
) -> float:
    """Gate direct classifier replay before either tensor leaves its device."""
    _require(logits_model.shape == logits_direct.shape, "same-backend linear shape drift")
    _require(logits_model.dtype == logits_direct.dtype, "same-backend linear dtype drift")
    _require(logits_model.device == logits_direct.device, "same-backend linear device drift")
    if require_cuda:
        _require(logits_model.is_cuda and logits_direct.is_cuda, "same-backend real gate is not on CUDA")
    _require(bool(torch.isfinite(logits_model).all()) and bool(torch.isfinite(logits_direct).all()),
             "same-backend linear output is nonfinite")
    error = float(torch.max(torch.abs(logits_direct - logits_model)).item()) if logits_model.numel() else 0.0
    _require(
        math.isfinite(error) and error <= SAME_BACKEND_TOLERANCE,
        f"same-backend direct linear replay exceeds 1e-6: {error}",
    )
    return error


def validate_saved_output_replay(arrays: Mapping[str, Any], *, np: Any) -> dict[str, float]:
    _require_exact_fields(arrays)
    logits = np.asarray(arrays["logits"])
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / np.sum(exp, axis=1, keepdims=True)
    errors = {
        "saved_Wz_plus_b_vs_logits_max_abs_error": _finite_max_abs(
            arrays["Wz_plus_b"], logits, np=np, name="saved Wz_plus_b/logits"
        ),
        "saved_softmax_max_abs_error": _finite_max_abs(
            arrays["probabilities"], softmax, np=np, name="saved softmax"
        ),
        "repeat_logits_max_abs_error": _finite_max_abs(
            arrays["repeat_logits"], logits, np=np, name="repeat logits"
        ),
        "repeat_z_max_abs_error": _finite_max_abs(
            arrays["repeat_z"], arrays["z"], np=np, name="repeat z"
        ),
    }
    violations = {name: value for name, value in errors.items() if value > STRICT_REPLAY_TOLERANCE}
    _require(not violations, f"saved-output replay exceeds 1e-6: {violations}")
    return errors


def _error_summary(observed: Any, expected: Any, *, np: Any) -> dict[str, float]:
    error = np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    _require(bool(np.isfinite(error).all()), "cross-backend diagnostic is nonfinite")
    return {
        "max_abs_error": float(np.max(error)) if error.size else 0.0,
        "p95_abs_error": float(np.percentile(error, 95)) if error.size else 0.0,
        "p99_abs_error": float(np.percentile(error, 99)) if error.size else 0.0,
    }


def cross_backend_diagnostics(arrays: Mapping[str, Any], *, np: Any, torch: Any) -> list[dict[str, Any]]:
    """Compute finite portability diagnostics without applying magnitude gates."""
    _require_exact_fields(arrays)
    z = np.array(arrays["z"], dtype=np.float32, order="C", copy=True)
    weight = np.array(arrays["classifier_weight"], dtype=np.float32, order="C", copy=True)
    bias = np.array(arrays["classifier_bias"], dtype=np.float32, order="C", copy=True)
    logits = np.asarray(arrays["logits"], dtype=np.float32)
    _require(
        bool(np.isfinite(z).all() and np.isfinite(weight).all() and np.isfinite(bias).all()
             and np.isfinite(logits).all()),
        "cross-backend diagnostic input is nonfinite",
    )
    cpu_torch = torch.nn.functional.linear(
        torch.from_numpy(z), torch.from_numpy(weight), torch.from_numpy(bias)
    ).detach().numpy()
    numpy32 = z @ weight.T + bias
    numpy64 = z.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)
    logit_abs = np.abs(logits.astype(np.float64))
    shared = {
        "feature_dimension": int(z.shape[1]),
        "logit_abs_max": float(np.max(logit_abs)) if logit_abs.size else 0.0,
        "logit_abs_p95": float(np.percentile(logit_abs, 95)) if logit_abs.size else 0.0,
        "diagnostic_only": True,
        "finite": True,
    }
    values = (
        ("CPU_PyTorch_float32", cpu_torch),
        ("NumPy_float32", numpy32),
        ("NumPy_float64", numpy64),
    )
    return [
        {"backend": name, **_error_summary(value, logits, np=np), **shared}
        for name, value in values
    ]


def _atomic_save_npz(path: Path, arrays: Mapping[str, Any], *, np: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def replay_persisted_artifact(
    path: str | Path,
    *,
    expected_digests: Mapping[str, Mapping[str, Any]],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"persisted target artifact is absent: {target}")
    with np.load(target, allow_pickle=False) as archive:
        _require(set(archive.files) == set(TARGET_ARRAY_FIELDS),
                 "persisted target artifact schema drift or target-label field present")
        arrays = {field: np.array(archive[field], copy=True) for field in TARGET_ARRAY_FIELDS}
    observed = verify_digest_registry(expected_digests, arrays, np=np)
    saved = validate_saved_output_replay(arrays, np=np)
    diagnostics = cross_backend_diagnostics(arrays, np=np, torch=torch)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA,
        "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "array_digests": observed,
        "saved_output_replay": saved,
        "cross_backend_diagnostics": diagnostics,
        "rows": int(np.asarray(arrays["logits"]).shape[0]),
    }


def write_and_replay_artifact(
    path: str | Path,
    *,
    arrays: Mapping[str, Any],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    _require_exact_fields(arrays)
    target = Path(path)
    _require(not target.exists(), "target artifact overwrite is forbidden")
    expected = build_digest_registry(arrays, np=np)
    validate_saved_output_replay(arrays, np=np)
    cross_backend_diagnostics(arrays, np=np, torch=torch)
    _atomic_save_npz(target, arrays, np=np)
    return replay_persisted_artifact(
        target, expected_digests=expected, np=np, torch=torch,
    )
