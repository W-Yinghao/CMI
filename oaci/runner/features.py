"""Canonical frozen-feature extraction for selection / audit leakage.

Rows are extracted in stable ``sample_id`` order; the model factory runs inside a forked RNG; the
loaded checkpoint is re-verified byte-exact; everything runs under per-submodule ``eval`` +
``inference_mode`` so the caller's RNG and the model state are unchanged. The produced
``FrozenFeatures`` carries the REAL string sample/group ids, and its population hash must equal the
expected ``LeakageDesign``'s.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ..leakage.crossfit import FrozenFeatures, feat_population_hash
from ..train.bn import all_eval
from ..train.checkpoint import model_state_hash
from ..train.rng import forked_rng


@dataclass(frozen=True)
class FeatureArtifact:
    features: FrozenFeatures
    feature_hash: str
    model_hash: str
    population_hash: str


def _feature_hash(population_hash, Z) -> str:
    a = np.ascontiguousarray(Z)
    h = hashlib.sha256()
    h.update(population_hash.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def extract_frozen_features(model_state, expected_model_hash, model_factory, training_data,
                            expected_design, *, factory_seed, chunk_size, device) -> FeatureArtifact:
    if training_data.d is None or training_data.group is None:
        raise ValueError("feature extraction needs domain and group ids on the training data")
    n = len(training_data)
    order = sorted(range(n), key=lambda i: training_data.sample_id[i])     # canonical sample-id order

    with forked_rng(factory_seed, device):                                 # factory must not touch caller RNG
        model = model_factory()
    model = model.to(device)
    model.load_state_dict(model_state)
    mh = model_state_hash(model)
    if mh != expected_model_hash:
        raise ValueError("model hash mismatch after loading the checkpoint for feature extraction")

    rng_before = torch.random.get_rng_state()
    cs = n if chunk_size is None else int(chunk_size)
    chunks = []
    with all_eval(model), torch.inference_mode():
        for a in range(0, n, cs):
            idx = torch.as_tensor(order[a:a + cs], dtype=torch.long)
            chunks.append(np.asarray(model(training_data.X[idx].to(device)).z.cpu(), dtype=np.float64))
    Z = np.concatenate(chunks, axis=0)

    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise RuntimeError("feature extraction perturbed the caller RNG")
    if model_state_hash(model) != mh:
        raise RuntimeError("feature extraction mutated the model state")

    y = training_data.y.detach().cpu().numpy()[order]
    d = training_data.d.detach().cpu().numpy()[order]
    mass = training_data.sample_mass.detach().cpu().numpy()[order]
    sid = tuple(str(training_data.sample_id[i]) for i in order)
    grp = tuple(str(training_data.group[i]) for i in order)
    feat = FrozenFeatures(Z=Z, y=y, d=d, group=grp, sample_mass=mass, sample_id=sid)

    pop = feat_population_hash(feat)
    if pop != expected_design.population_hash:
        raise ValueError("extracted feature population hash != expected LeakageDesign population hash")
    return FeatureArtifact(features=feat, feature_hash=_feature_hash(pop, Z), model_hash=mh, population_hash=pop)
