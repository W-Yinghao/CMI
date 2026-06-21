"""Audit-infra unit tests (fast, no training): experiment/shard signatures, source-training
provenance bundles, exact-key shard merge, legacy-output + git-state guards."""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from h2cmi.config import H2Config, core_config
from h2cmi import grid_io as gio
from h2cmi.grid_io import (config_signature, source_training_signature, source_data_hash,
                           build_manifest, validate_or_create_manifest, manifest_path,
                           save_source_bundle, load_source_bundle, source_bundle_paths,
                           variant_id, require_clean_git, global_expected_keys)
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.train.trainer import H2Model
from h2cmi.merge_grid_shards import merge_shards

GLOBAL = dict(global_seeds=[0, 1, 2], global_sites=[0, 1, 2, 3, 4], scenarios=["cov", "prior"],
              items=["identity", "joint"], item_field="action", cmi_arms=["off", "on"])


def _cfg():
    c = core_config(H2Config(n_classes=3))
    c.encoder.n_chans = 8; c.encoder.n_times = 64
    return c


# ---- signatures ----
def test_config_signature_covers_full_config():
    a = _cfg()
    for mut in (lambda c: setattr(c.tta, "em_iters", 999),
                lambda c: setattr(c.density, "cov_rank", 9),
                lambda c: setattr(c.train, "lr", 0.001234),
                lambda c: setattr(c.encoder, "n_chans", 99)):
        b = _cfg(); mut(b)
        assert config_signature(a) != config_signature(b)
    assert config_signature(a) == config_signature(_cfg())


def test_source_training_signature_ignores_tta_not_source():
    a = _cfg()
    t = _cfg(); t.tta.em_iters = 99; t.tta.trust_region = 5.0   # TTA must NOT invalidate a bundle
    assert source_training_signature(a, 0, 0, "off") == source_training_signature(t, 0, 0, "off")
    e = _cfg(); e.encoder.z_c_dim = 99                          # source config must invalidate
    assert source_training_signature(a, 0, 0, "off") != source_training_signature(e, 0, 0, "off")
    assert source_training_signature(a, 0, 0, "off") != source_training_signature(a, 1, 0, "off")
    assert source_training_signature(a, 0, 0, "off") != source_training_signature(a, 0, 0, "on")


def test_variant_id_normalised():
    assert variant_id(resp="gen", update="oneshot", alpha=0.5, prior=None) == \
        "alpha=0.5__resp=gen__update=oneshot"


# ---- experiment vs shard ----
def test_real_shard_invocations_share_experiment_signature():
    m0 = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [1]}, cli={}, **GLOBAL)
    m1 = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [2]}, cli={}, **GLOBAL)
    assert m0["experiment_signature"] == m1["experiment_signature"]
    assert m0["shard_spec"] != m1["shard_spec"]
    # different GLOBAL grid -> different experiment
    g2 = dict(GLOBAL); g2["global_sites"] = [0, 1, 2, 3]
    m2 = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [1]}, cli={}, **g2)
    assert m2["experiment_signature"] != m0["experiment_signature"]


def test_manifest_guard_and_legacy_abort():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "g.jsonl")
        m = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [0]}, cli={}, **GLOBAL)
        validate_or_create_manifest(out, m)
        validate_or_create_manifest(out, m)                       # idempotent on match
        c2 = _cfg(); c2.tta.em_iters = 5
        m2 = build_manifest(c2, shard_spec={"seeds": [0], "sites": [0]}, cli={}, **GLOBAL)
        try:
            validate_or_create_manifest(out, m2); assert False
        except RuntimeError:
            pass
    # non-empty output with NO manifest must abort (no auto-adoption)
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "legacy.jsonl")
        with open(out, "w") as f:
            f.write('{"data_seed":0}\n')
        try:
            validate_or_create_manifest(out, m); assert False
        except RuntimeError:
            pass


# ---- source-data hash includes domains/DAG ----
def _domains(levels, parents=("site",)):
    dag = DomainDAG([DomainFactor("site", 3, (), "invariant", 0.02),
                     DomainFactor("subject", 6, ("site",), "random_effect", 0.05)])
    return DomainLabels(dag, levels)


def test_bundle_hash_includes_domains_and_dag():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 4, 8)).astype("float32"); y = rng.integers(0, 3, 12)
    lv = np.stack([rng.integers(0, 3, 12), rng.integers(0, 6, 12)], 1)
    lv2 = lv.copy(); lv2[0, 1] = (lv2[0, 1] + 1) % 6           # change ONE domain level
    h1 = source_data_hash(X, y, _domains(lv))
    h2 = source_data_hash(X, y, _domains(lv2))
    assert h1 != h2, "source_data_hash ignored domain levels"
    assert h1 == source_data_hash(X, y, _domains(lv)), "not stable"


# ---- source bundle provenance ----
def test_bundle_records_and_validates_source_provenance():
    cfg = _cfg(); pi = np.full(3, 1 / 3); model = H2Model(cfg, pi)
    tsig = source_training_signature(cfg, 0, 1, "off")
    with tempfile.TemporaryDirectory() as d:
        pt, js = source_bundle_paths(d, tsig, 0, 1, "off")
        save_source_bundle(pt, js, model, training_signature=tsig, source_data_hash="abc",
                           pi_star=pi, commit_sha="deadbeef", history=[{"epoch": 0}])
        meta = json.load(open(js))
        assert meta["source_training_signature"] == tsig
        assert meta["source_training_commit_sha"] == "deadbeef"
        m2, _ = load_source_bundle(pt, js, build_model=lambda: H2Model(cfg, pi),
                                   expected_training_signature=tsig, expected_source_data_hash="abc")
        for (k, a), (_, b) in zip(model.state_dict().items(), m2.state_dict().items()):
            assert torch.allclose(a, b)
        for bad in (dict(expected_training_signature="X", expected_source_data_hash="abc"),
                    dict(expected_training_signature=tsig, expected_source_data_hash="X")):
            try:
                load_source_bundle(pt, js, build_model=lambda: H2Model(cfg, pi), **bad); assert False
            except RuntimeError:
                pass
        # tamper the stored checkpoint hash -> abort
        meta["source_checkpoint_hash"] = "0" * 64
        json.dump(meta, open(js, "w"))
        try:
            load_source_bundle(pt, js, build_model=lambda: H2Model(cfg, pi),
                               expected_training_signature=tsig, expected_source_data_hash="abc")
            assert False
        except RuntimeError:
            pass


# ---- shard merge ----
def _shard(path, manifest, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(manifest_path(path), "w") as f:
        json.dump(manifest, f)


def _row(seed, site, action, cmi, scen="cov"):
    return dict(data_seed=seed, target_site=site, scenario=scen, action=action, cmi=cmi)


def test_merge_exact_keys_and_writes_manifest():
    g = dict(global_seeds=[0], global_sites=[0, 1], scenarios=["cov"],
             items=["identity", "joint"], item_field="action", cmi_arms=["off"])
    man0 = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [0]}, cli={}, **g)
    man1 = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [1]}, cli={}, **g)
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "shards"; sd.mkdir()
        _shard(str(sd / "s0.jsonl"), man0, [_row(0, 0, "identity", "off"), _row(0, 0, "joint", "off")])
        _shard(str(sd / "s1.jsonl"), man1, [_row(0, 1, "identity", "off"), _row(0, 1, "joint", "off")])
        out = str(Path(d) / "merged.jsonl")
        info = merge_shards(str(sd), out, item_field="action")
        assert info["rows"] == 4 and info["unique_keys"] == 4
        assert Path(manifest_path(out)).exists(), "merge did not write a merged manifest"


def test_merge_rejects_missing_manifest():
    g = dict(GLOBAL);
    man = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [0]}, cli={}, **g)
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "shards"; sd.mkdir()
        # shard WITHOUT a manifest
        with open(sd / "s0.jsonl", "w") as f:
            f.write(json.dumps(_row(0, 0, "identity", "off")) + "\n")
        try:
            merge_shards(str(sd), str(Path(d) / "m.jsonl"), item_field="action"); assert False
        except RuntimeError:
            pass


def test_merge_rejects_wrong_key_with_correct_count():
    g = dict(global_seeds=[0], global_sites=[0, 1], scenarios=["cov"],
             items=["identity", "joint"], item_field="action", cmi_arms=["off"])
    man = build_manifest(_cfg(), shard_spec={"seeds": [0], "sites": [0, 1]}, cli={}, **g)
    expected = global_expected_keys(man)
    assert len(expected) == 4
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "shards"; sd.mkdir()
        # right count (4) but ONE key wrong: (0,1,cov,joint,off) replaced by (0,1,cov,joint,ON)
        _shard(str(sd / "s.jsonl"), man, [_row(0, 0, "identity", "off"), _row(0, 0, "joint", "off"),
                                          _row(0, 1, "identity", "off"), _row(0, 1, "joint", "on")])
        try:
            merge_shards(str(sd), str(Path(d) / "m.jsonl"), item_field="action"); assert False
        except ValueError:
            pass


# ---- git guard ----
def test_unknown_or_dirty_git_state_aborts(monkeypatch=None):
    orig = gio.git_state
    try:
        gio.git_state = lambda: ("unknown", True)
        try:
            require_clean_git(); assert False
        except RuntimeError:
            pass
        gio.git_state = lambda: ("abc123", True)
        try:
            require_clean_git(); assert False
        except RuntimeError:
            pass
        assert require_clean_git(allow_dirty=True) == "abc123"   # dev escape
        gio.git_state = lambda: ("abc123", False)
        assert require_clean_git() == "abc123"
    finally:
        gio.git_state = orig


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_grid_io PASSED")
