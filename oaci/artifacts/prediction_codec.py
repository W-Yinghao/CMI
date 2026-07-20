"""Codecs for PredictionBundle, EvaluationMetrics, leakage results and training diagnostics.

Predictions store arrays in NPZ + metadata in canonical JSON and rebuild the PredictionBundle, then
re-verify every hash. Metrics / leakage / diagnostics are canonical JSON only (no pickle) and their
scientific hash is recomputed on read.
"""
from __future__ import annotations

import numpy as np

from ..eval.artifacts import PredictionBundle
from ..runner.metrics import EvaluationMetrics, metrics_payload
from ..runner.scientific_hash import leakage_result_hash, normalize_keys, scientific_value_hash
from .deterministic_npz import to_unicode_array

PREDICTION_KIND, METRICS_KIND, LEAKAGE_KIND, DIAGNOSTICS_KIND = (
    "prediction", "metrics", "leakage", "diagnostics")


# ---------------------------------- prediction bundle ----------------------------------
def encode_prediction(b: PredictionBundle) -> tuple:
    body = {"method": b.method, "seed": int(b.seed), "split_id": b.split_id, "split_role": b.split_role,
            "deletion_level": int(b.deletion_level), "class_names": [str(c) for c in b.class_names],
            "risk_metric": b.risk_metric, "support_mask_hash": b.support_mask_hash,
            "checkpoint_hash": b.checkpoint_hash, "input_tensor_hash": b.input_tensor_hash,
            "split_manifest_hash": b.split_manifest_hash, "preprocess_hash": b.preprocess_hash,
            "eval_population_hash": b.eval_population_hash, "bundle_hash": b.bundle_hash,
            "audit_signature_hash": b.audit_signature_hash, "prediction_content_hash": b.prediction_content_hash()}
    arrays = {"sample_id": to_unicode_array(b.sample_id.tolist()),
              "logits": np.ascontiguousarray(b.logits, dtype=np.float64),
              "y": np.ascontiguousarray(b.y, dtype=np.int64), "domain": np.ascontiguousarray(b.domain, dtype=np.int64),
              "group": to_unicode_array(b.group.tolist())}
    return b.prediction_content_hash(), body, arrays


def decode_prediction(body, arrays) -> PredictionBundle:
    b = PredictionBundle(
        sample_id=arrays["sample_id"], logits=arrays["logits"], y=arrays["y"], domain=arrays["domain"],
        group=arrays["group"], method=body["method"], seed=int(body["seed"]), split_id=body["split_id"],
        split_role=body["split_role"], deletion_level=int(body["deletion_level"]),
        class_names=tuple(body["class_names"]), risk_metric=body["risk_metric"],
        support_mask_hash=body["support_mask_hash"], checkpoint_hash=body["checkpoint_hash"],
        audit_tensor_hash=body["input_tensor_hash"], split_manifest_hash=body["split_manifest_hash"],
        preprocess_hash=body["preprocess_hash"])
    if (b.eval_population_hash != body["eval_population_hash"] or b.bundle_hash != body["bundle_hash"]
            or b.audit_signature_hash != body["audit_signature_hash"]
            or b.prediction_content_hash() != body["prediction_content_hash"]):
        raise ValueError("decoded prediction bundle hashes do not match the stored metadata")
    return b


# ---------------------------------- metrics ----------------------------------
def encode_metrics(em: EvaluationMetrics) -> tuple:
    pay = metrics_payload(em)
    body = {**pay, "domain_class_coverage_items": [[int(d), float(c)] for d, c in em.domain_class_coverage_items],
            "metrics_hash": em.metrics_hash}
    return em.metrics_hash, body, None


def decode_metrics(body) -> EvaluationMetrics:
    f = dict(body)
    cov = tuple((int(d), float(c)) for d, c in f["domain_class_coverage_items"])
    em = EvaluationMetrics(**{k: v for k, v in f.items() if k not in ("metrics_hash", "domain_class_coverage_items")},
                           domain_class_coverage_items=cov, metrics_hash=f["metrics_hash"])
    if scientific_value_hash(metrics_payload(em)) != em.metrics_hash:
        raise ValueError("decoded metrics hash does not recompute")
    return em


# ---------------------------------- leakage ----------------------------------
def encode_leakage(leak) -> tuple:
    logical = leakage_result_hash(leak)
    return logical, normalize_keys(leak), None


def decode_leakage(body, logical_hash) -> dict:
    if scientific_value_hash(body) != logical_hash:
        raise ValueError("decoded leakage hash does not recompute")
    return body


# ---------------------------------- diagnostics ----------------------------------
def encode_diagnostics(diag) -> tuple:
    d = dict(diag)
    return scientific_value_hash(d), d, None


def decode_diagnostics(body, logical_hash) -> dict:
    if scientific_value_hash(body) != logical_hash:
        raise ValueError("decoded diagnostics hash does not recompute")
    return body
