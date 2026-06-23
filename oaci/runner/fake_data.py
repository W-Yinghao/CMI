"""Deterministic two-level FAKE fixture (A2b-2b-i).

A fully synthetic population with exact row/unit counts and support tables, built from a strict
``status: smoke`` ``FAKE_TWO_LEVEL`` manifest. Every feature component is derived from a stable id via
``derive_seed`` (never the global RNG), so the FoldData is independent of row order, model seed and
method order; only ``fake_fixture.data_seed`` moves the tensor hash.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ..models import build_model
from ..protocol.manifest_v2 import load_v2
from ..train.rng import derive_seed
from .config import ModelSpec, RunnerExecutionConfig
from .data import FoldData
from .keys import FoldKey, canonical_json_hash
from .maps import build_frozen_maps
from .scope import ScopePlanConfig, build_fold_scope
from .support import DeletionCell, make_deletion_schedule

_DS = "FAKE_TWO_LEVEL"
_CLASSES = ("c0", "c1")


@dataclass(frozen=True)
class FakeFold:
    manifest: object
    manifest_payload: dict
    manifest_hash: str
    fold_data: FoldData
    maps: object
    deletion_schedule: object
    fold_scope: object
    scope_config: object
    execution_config: RunnerExecutionConfig
    model_spec: ModelSpec
    fake_data_hash: str

    def model_factory(self):
        ff = self.manifest.fake_fixture
        z, h, d = int(self.manifest.backbone.mlp_z_dim), int(self.manifest.backbone.mlp_hidden), int(ff.input_dim)
        return lambda: build_model("mlp", in_dim=d, n_classes=2, z_dim=z, hidden=h)


def _unit_vec(data_seed, namespace, stable_id, dim):
    rng = np.random.default_rng(derive_seed(int(data_seed), namespace, str(stable_id)))
    v = rng.standard_normal(dim).astype(np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rows(ff):
    """Build the canonical (role, domain, recording, class, window) rows with cycled window counts."""
    cycle = list(ff.windows_per_unit_cycle)
    roles = [("source_train", list(ff.source_domain_ids), int(ff.source_train_groups_per_domain)),
             ("source_audit", list(ff.source_domain_ids), int(ff.source_audit_groups_per_domain)),
             ("target_audit", list(ff.target_domain_ids), int(ff.target_groups_per_domain))]
    rows = []
    for role, domains, n_groups in roles:
        unit_k = 0
        for dom in domains:                                    # canonical: domain, recording, class
            for r in range(n_groups):
                rec = f"{dom}-rec{r}"
                for c in (0, 1):
                    n_win = cycle[unit_k % len(cycle)]; unit_k += 1
                    unit = f"{_DS}|{role}|{dom}|{rec}|c{c}"
                    grp = f"{_DS}|{role}|{dom}|{rec}"
                    mass = 1.0 / n_win
                    for w in range(n_win):
                        sid = f"{_DS}|{role}|{dom}|{rec}|c{c}|window-{w}"
                        rows.append(dict(sid=sid, role=role, dom=dom, grp=grp, unit=unit, y=c, mass=mass))
    return rows


def _build_X(rows, ff):
    dim = int(ff.input_dim)
    a_y, a_d, a_g, a_e = (float(ff.class_signal_scale), float(ff.domain_signal_scale),
                          float(ff.recording_signal_scale), float(ff.window_noise_scale))
    vy = {c: _unit_vec(ff.data_seed, "fake_class_signal", f"c{c}", dim) for c in (0, 1)}
    vd = {d: _unit_vec(ff.data_seed, "fake_domain_signal", d, dim) for d in {r["dom"] for r in rows}}
    vg = {g: _unit_vec(ff.data_seed, "fake_recording_effect", g, dim) for g in {r["grp"] for r in rows}}
    X = np.zeros((len(rows), dim), dtype=np.float64)
    for i, r in enumerate(rows):
        eps = _unit_vec(ff.data_seed, "fake_window_noise", r["sid"], dim)
        X[i] = a_y * vy[r["y"]] + a_d * vd[r["dom"]] + a_g * vg[r["grp"]] + a_e * eps
    return torch.from_numpy(np.ascontiguousarray(X.astype(np.float32)))


def build_fake_fold(manifest_path) -> FakeFold:
    m = load_v2(manifest_path); m.validate_complete()
    if m.fake_fixture is None:
        raise ValueError("not a fake-fixture manifest")
    ff = m.fake_fixture
    payload = {"canonical_json": m.to_canonical_json()}
    manifest_hash = m.freeze()["sha256"]

    rows = _rows(ff)
    ri = {"source_train": [], "source_audit": [], "target_audit": []}
    for i, r in enumerate(rows):
        ri[r["role"]].append(i)
    X = _build_X(rows, ff)
    preprocess_hash = "fake_preprocess:" + canonical_json_hash(
        {"kind": "deterministic_synthetic_identity", "data_seed": int(ff.data_seed), "input_dim": int(ff.input_dim),
         "scales": [ff.class_signal_scale, ff.domain_signal_scale, ff.recording_signal_scale, ff.window_noise_scale]})
    split_manifest_hash = "fake_split:" + hashlib.sha256(
        "|".join(f"{r['role']}::{r['sid']}" for r in sorted(rows, key=lambda r: r["sid"])).encode()).hexdigest()

    fd = FoldData.from_arrays(
        X=X, y=np.array([r["y"] for r in rows]), sample_id=[r["sid"] for r in rows],
        domain_id=[r["dom"] for r in rows], group_id=[r["grp"] for r in rows],
        support_unit_id=[r["unit"] for r in rows], mass_unit_id=[r["unit"] for r in rows],
        eval_unit_id=[r["unit"] for r in rows], sample_mass=np.array([r["mass"] for r in rows]),
        class_names=list(_CLASSES), source_train_idx=np.array(ri["source_train"]),
        source_audit_idx=np.array(ri["source_audit"]), target_audit_idx=np.array(ri["target_audit"]),
        preprocess_hash=preprocess_hash, split_manifest_hash=split_manifest_hash, preprocess_fit_ids=frozenset())

    eval_domains = sorted({r["dom"] for r in rows if r["role"] in ("source_audit", "target_audit")})
    maps = build_frozen_maps(list(_CLASSES), list(ff.source_domain_ids), eval_domains)
    schedule = make_deletion_schedule([DeletionCell("S0", "c1")], fd, maps)
    cfg = ScopePlanConfig.from_manifest(m, support_m=int(m.enabled_datasets()[_DS].support_m))
    fold_key = FoldKey(manifest_hash, _DS, "f0", int(m.seeds.split), int(m.seeds.deletion))
    fold_scope = build_fold_scope(fold_key, maps, fd, schedule, cfg)
    exec_cfg = RunnerExecutionConfig.from_manifest(m)
    model_spec = ModelSpec.build("mlp", {"z_dim": int(m.backbone.mlp_z_dim), "hidden": int(m.backbone.mlp_hidden)},
                                 (int(ff.input_dim),), 2)

    fake = FakeFold(manifest=m, manifest_payload=payload, manifest_hash=manifest_hash, fold_data=fd, maps=maps,
                    deletion_schedule=schedule, fold_scope=fold_scope, scope_config=cfg, execution_config=exec_cfg,
                    model_spec=model_spec, fake_data_hash=fd.data_contract_hash)
    _assert_fixture(fake, rows)
    return fake


def _assert_fixture(fake: FakeFold, rows) -> None:
    fd = fake.fold_data
    counts = {role: (len({r["unit"] for r in rows if r["role"] == role}),
                     sum(1 for r in rows if r["role"] == role)) for role in
              ("source_train", "source_audit", "target_audit")}
    if counts != {"source_train": (24, 48), "source_audit": (18, 36), "target_audit": (8, 15)}:
        raise AssertionError(f"fake row/unit counts wrong: {counts}")
    for unit in {r["unit"] for r in rows}:                      # every mass unit sums to exactly 1
        if abs(sum(r["mass"] for r in rows if r["unit"] == unit) - 1.0) > 1e-9:
            raise AssertionError(f"mass unit {unit} != 1")
    fd.assert_integrity()
