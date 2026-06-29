"""REVIEW_P0 shared 3-seed source loader with STRICT provenance validation.

seed 0: reuse the frozen bundle ONLY after validating code_signature + recomputed source data_hash +
source-training config (epochs, n_chans, n_times). On any mismatch raise ProvenanceError (the caller
STOPS that unit and reports -- provenance is never silently relaxed). seeds 1,2: train fresh into a NEW
bundle root (the frozen seed-0 artifacts are never overwritten). Returns
(model, pooled_ref=(mu,sd), R_src, pi_star, validated_seed0: bool).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from h2cmi.config import core_config, H2Config
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import reference_weighted_source_moments
from h2cmi.eval.ea import reference_cov


class ProvenanceError(Exception):
    pass


def data_hash(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def source_sig(tag, code_sig, cfg):
    s = f"{tag}|{code_sig}|ep{cfg.train.epochs}|ch{cfg.encoder.n_chans}|t{cfg.encoder.n_times}|sd{cfg.train.seed}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _site_domains(subject):
    subs = np.unique(subject); smap = {int(s): i for i, s in enumerate(subs)}
    site = np.array([smap[int(s)] for s in subject], np.int64)
    dag = DomainDAG([DomainFactor("site", max(1, len(subs)), (), "invariant", 0.02)])
    return dag, DomainLabels(dag, site.reshape(-1, 1))


def get_source_p0(seed0_root, new_root, tag, cfg, code_sig, K, data_fn):
    """data_fn() -> (X, y, subject). cfg.train.seed selects the seed. Cheap (in-memory slicing); used for
    BOTH validation and (cache-miss) training -- never forces a reload of an already-trained model."""
    seed = cfg.train.seed
    sig = source_sig(tag, code_sig, cfg)
    root = seed0_root if seed == 0 else new_root
    os.makedirs(new_root, exist_ok=True)
    pt = os.path.join(root, f"{sig}.pt"); js = os.path.join(root, f"{sig}.json")
    npz = os.path.join(root, f"{sig}.moments.npz")
    pi_unif = np.full(K, 1.0 / K)
    X, y, subject = data_fn()
    dh = data_hash(X, y)
    if os.path.exists(pt) and os.path.exists(js):
        meta = json.load(open(js))
        # STRICT: reject missing (None) fields -- no permissive None-tolerance (review #2B).
        for f in ("code_sig", "data_hash", "epochs", "n_chans"):
            if meta.get(f) is None:
                raise ProvenanceError(f"{tag} seed{seed}: sidecar field '{f}' is null (strict provenance)")
        if meta.get("code_sig") != code_sig:
            raise ProvenanceError(f"{tag} seed{seed}: code_sig {meta.get('code_sig')} != {code_sig}")
        if meta.get("data_hash") != dh:
            raise ProvenanceError(f"{tag} seed{seed}: data_hash {meta.get('data_hash')} != {dh}")
        if meta.get("epochs") != cfg.train.epochs or meta.get("n_chans") != cfg.encoder.n_chans:
            raise ProvenanceError(f"{tag} seed{seed}: source-training config mismatch")
        model = H2Model(cfg, pi_unif).to(cfg.train.device)
        model.load_state_dict(torch.load(pt, map_location=cfg.train.device)); model.eval()
        pi_star = reference_prior(y, K, "uniform")
        if os.path.exists(npz):
            m = np.load(npz); pooled_ref = (torch.tensor(m["mu"], dtype=torch.float32), torch.tensor(m["sd"], dtype=torch.float32)); R_src = m["R_src"]
        else:
            Us = _embed(model, X, cfg.train.device); pooled_ref = reference_weighted_source_moments(Us, y, pi_star); R_src = reference_cov(X)
        return model, pooled_ref, R_src, pi_star, (seed == 0)
    # cache miss -> train (seed 1/2, or a seed-0 unit that was never run)
    pi_star = reference_prior(y, K, "uniform")
    dag, dom = _site_domains(subject)
    model, *_ = train_h2(X, y, dom, dag, cfg, align_factor="site")
    torch.save(model.state_dict(), os.path.join(new_root, f"{sig}.pt"))
    Us = _embed(model, X, cfg.train.device)
    mu, sd = reference_weighted_source_moments(Us, y, pi_star); R_src = reference_cov(X)
    np.savez(os.path.join(new_root, f"{sig}.moments.npz"), mu=np.asarray(mu), sd=np.asarray(sd), R_src=R_src)
    json.dump(dict(tag=tag, code_sig=code_sig, data_hash=dh, sig=sig, seed=seed,
                   epochs=cfg.train.epochs, n_chans=cfg.encoder.n_chans, n_train=int(len(y))),
              open(os.path.join(new_root, f"{sig}.json"), "w"), indent=2)
    return model, (mu, sd), R_src, pi_star, False
