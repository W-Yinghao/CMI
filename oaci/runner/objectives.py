"""Fresh objective per (method, run). OACIObjective accumulates per-row diagnostics, so it must NOT
be reused across method order / levels / repeats — this factory always returns a new instance.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..methods import GlobalLPCObjective, OACIObjective, UniformObjective
from .keys import feed_int64, feed_string
from .results import ObjectiveSpec


def _spec(method, active, reason, support_hash, prior) -> ObjectiveSpec:
    h = hashlib.sha256(); feed_string(h, method)
    h.update(b"1" if active else b"0"); feed_string(h, reason or "-"); feed_string(h, support_hash)
    if prior is not None:
        p = np.ascontiguousarray(np.asarray(prior, dtype=np.float64))
        h.update(str(p.shape).encode()); h.update(p.tobytes())
        prior = np.array(p, copy=True); prior.setflags(write=False)
    return ObjectiveSpec(method, active, reason, support_hash, prior, h.hexdigest())


def _check_prior(prior, n_classes, n_d0):
    if prior.shape != (n_classes, n_d0):
        raise ValueError(f"prior matrix shape {prior.shape} != ({n_classes}, {n_d0})")
    if not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("prior matrix must be finite and strictly positive")
    if not np.allclose(prior.sum(axis=1), 1.0):
        raise ValueError("each prior row must sum to 1")


def make_objective(method_name, support_state, fold_scope, execution_cfg):
    maps = fold_scope.maps
    nc, nd0 = len(maps.class_names), len(maps.source_domain_ids)
    status = dict(support_state.method_status_items)[method_name]
    sup = support_state.support_hash

    if method_name == "ERM":
        return None, _spec("ERM", True, None, sup, None)

    if method_name == "OACI":
        obj = OACIObjective(support_graph=support_state.support_graph, adv_hidden=execution_cfg.method_critic_hidden)
        if obj.active_status().active != status.active:
            raise ValueError("OACI objective activity != runner MethodStatus")
        return obj, _spec("OACI", status.active, status.reason, sup, None)

    cell_mass = np.asarray(support_state.cell_mass, dtype=np.float64)
    class_mass = cell_mass.sum(axis=0)
    present = list(range(nc))
    if method_name == "global_lpc":
        obj = GlobalLPCObjective(level0_domains=list(range(nd0)), cell_mass=cell_mass, class_mass=class_mass,
                                 reference_prior=fold_scope.level0_reference_prior, present_classes=present,
                                 alpha=execution_cfg.global_lpc_alpha, active=status.active,
                                 inactive_reason=status.reason, hidden=execution_cfg.method_critic_hidden)
        prior = np.stack([obj.prior_vector(y) for y in range(nc)])
        _check_prior(prior, nc, nd0)
        for c in support_state.deleted_cells:                         # deleted cell: 0 rows but positive prior
            d, yy = maps.source_domain_to_index[c.domain_id], maps.class_to_index[c.class_name]
            if cell_mass[d, yy] != 0:
                raise ValueError("deleted cell still has observed mass")
            if prior[yy, d] <= 0:
                raise ValueError("deleted cell prior mass must be > 0")
        return obj, _spec("global_lpc", status.active, status.reason, sup, prior)

    if method_name == "uniform":
        obj = UniformObjective(level0_domains=list(range(nd0)), class_mass=class_mass,
                               reference_prior=fold_scope.level0_reference_prior, present_classes=present,
                               active=status.active, inactive_reason=status.reason,
                               hidden=execution_cfg.method_critic_hidden)
        prior = np.stack([obj.prior_vector(y) for y in range(nc)])
        if not np.allclose(prior, 1.0 / nd0):
            raise ValueError("uniform prior must be exactly 1/|D0|")
        return obj, _spec("uniform", status.active, status.reason, sup, prior)

    raise ValueError(f"unknown method {method_name!r}")
