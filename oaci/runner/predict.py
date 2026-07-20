"""Eval-unit predictions for the three roles (A2b-1b-ii-b).

A checkpoint is forwarded once per (role, unique model hash) -- a forward-only operation under
eval() + inference_mode() that restores the caller RNG and re-verifies the model state. Window logits
are aggregated to eval units by mass-weighted mean posterior (with the manifest probability floor) and
wrapped in a PredictionBundle whose domain is the frozen contiguous int. The target role is forward
ONLY: it never fits anything.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from ..data.eeg.units import aggregate_mean_prob
from ..eval.artifacts import PredictionBundle, _feed_arr, _feed_strs
from ..train.bn import all_eval
from ..train.checkpoint import model_state_hash
from ..train.rng import forked_rng


@dataclass(frozen=True)
class RowPredictionArtifact:
    sample_id: tuple
    logits: np.ndarray
    model_hash: str
    role: str
    population_hash: str
    tensor_hash: str
    content_hash: str


def _content_hash(sample_id, logits, model_hash, role, population_hash, tensor_hash) -> str:
    sid = [str(s) for s in sample_id]
    order = sorted(range(len(sid)), key=lambda i: sid[i])
    h = hashlib.sha256()
    _feed_strs(h, [sid[i] for i in order]); _feed_arr(h, logits[order])
    _feed_strs(h, [model_hash, role, population_hash, tensor_hash])
    return h.hexdigest()


def predict_checkpoint(model_state, expected_model_hash, model_factory, role_view, *, factory_seed,
                       chunk_size, device) -> RowPredictionArtifact:
    sid = list(role_view.sample_id)
    if sid != sorted(sid):
        raise ValueError("role view sample ids must be canonical-sorted")
    n = len(sid)
    rng_entry = torch.random.get_rng_state()
    try:
        with forked_rng(factory_seed, device):                             # factory must not touch caller RNG
            model = model_factory()
        model = model.to(device)
        model.load_state_dict(model_state)
        mh = model_state_hash(model)
        if mh != expected_model_hash:
            raise ValueError("model hash mismatch after loading the checkpoint for prediction")
        cs = n if chunk_size is None else int(chunk_size)
        chunks = []
        with all_eval(model), torch.inference_mode():
            for a in range(0, n, cs):
                xb = role_view.X[a:a + cs].to(device)
                chunks.append(np.asarray(model(xb).logits.cpu(), dtype=np.float64))
        logits = np.concatenate(chunks, axis=0)
    finally:
        torch.random.set_rng_state(rng_entry)                              # caller RNG byte-exact regardless
    if model_state_hash(model) != mh:
        raise RuntimeError("prediction mutated the model state")
    if logits.ndim != 2 or logits.shape[0] != n:
        raise ValueError("logits must be [N, C] aligned to the role view")
    if not np.all(np.isfinite(logits)):
        raise ValueError("prediction logits must be finite")
    logits = np.ascontiguousarray(logits); logits.setflags(write=False)
    ch = _content_hash(role_view.sample_id, logits, mh, role_view.role,
                       role_view.population_hash, role_view.tensor_hash)
    return RowPredictionArtifact(sample_id=tuple(role_view.sample_id), logits=logits, model_hash=mh,
                                 role=role_view.role, population_hash=role_view.population_hash,
                                 tensor_hash=role_view.tensor_hash, content_hash=ch)


@dataclass(frozen=True)
class PredictionCacheKey:
    model_hash: str
    role: str
    population_hash: str
    tensor_hash: str
    model_spec_hash: str
    prediction_chunk_size: int | None


class RowPredictionCache:
    """One forward per (role, unique model hash). Different roles have different population/tensor
    identity, so a forward result is never reused across roles."""

    def __init__(self):
        self._store = {}
        self._req = defaultdict(int); self._comp = defaultdict(int); self._hit = defaultdict(int)

    def get_or_compute(self, key: PredictionCacheKey, fn) -> RowPredictionArtifact:
        self._req[key] += 1
        if key in self._store:
            self._hit[key] += 1
            return self._store[key]
        from .replay_store import resolve_artifact
        self._comp[key] += 1
        self._store[key] = resolve_artifact(f"logits:{key.role}", key, fn)   # role-segregated (C4b)
        return self._store[key]

    def total_requests(self): return int(sum(self._req.values()))
    def total_computes(self): return int(sum(self._comp.values()))
    def total_hits(self): return int(sum(self._hit.values()))

    def role_stats(self, role: str) -> tuple:
        req = sum(v for k, v in self._req.items() if k.role == role)
        comp = sum(v for k, v in self._comp.items() if k.role == role)
        hit = sum(v for k, v in self._hit.items() if k.role == role)
        return int(req), int(comp), int(hit)


def _validate_eval_units(eval_unit_id, y, domain_id, group_id) -> None:
    """Each eval unit must have a constant (label, stable domain id, stable group id)."""
    seen = {}
    for i, u in enumerate(eval_unit_id):
        key = (int(y[i]), str(domain_id[i]), str(group_id[i]))
        if u in seen and seen[u] != key:
            raise ValueError(f"eval unit {u!r} spans {seen[u]} and {key}")
        seen[u] = key


def aggregate_role_to_bundle(row_predictions, role_view, *, method_name, selected_model_hash, domain_map,
                             class_names, model_seed, fold_key_hash, support_hash, split_manifest_hash,
                             preprocess_hash, risk_metric, prob_floor, deletion_level) -> PredictionBundle:
    if tuple(row_predictions.sample_id) != tuple(role_view.sample_id):
        raise ValueError("row predictions and role view sample ids disagree")
    eu, y = role_view.eval_unit_id, role_view.y
    dom_s, grp = role_view.domain_id, role_view.group_id
    _validate_eval_units(eu, y, dom_s, grp)
    unit_ids, agg_logits, rep = aggregate_mean_prob(row_predictions.logits, eu,
                                                    prob_floor=float(prob_floor), sample_mass=role_view.sample_mass)
    unit_dom = np.array([int(domain_map[str(dom_s[r])]) for r in rep], dtype=np.int64)
    unit_grp = np.array([str(grp[r]) for r in rep], dtype=object)
    unit_y = np.array([int(y[r]) for r in rep], dtype=np.int64)
    return PredictionBundle(
        sample_id=np.array([str(u) for u in unit_ids.tolist()], dtype=object), logits=agg_logits, y=unit_y,
        domain=unit_dom, group=unit_grp, method=method_name, seed=int(model_seed), split_id=fold_key_hash,
        split_role=role_view.role, deletion_level=int(deletion_level), class_names=tuple(class_names),
        risk_metric=risk_metric, support_mask_hash=support_hash, checkpoint_hash=selected_model_hash,
        audit_tensor_hash=role_view.tensor_hash, split_manifest_hash=split_manifest_hash, preprocess_hash=preprocess_hash)
