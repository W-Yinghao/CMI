"""Audit-infra unit tests (fast, no training): config signature, manifest guard, source
bundle save/load verification, and strict shard merge."""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from h2cmi.config import H2Config, core_config
from h2cmi.grid_io import (config_signature, build_manifest, validate_or_create_manifest,
                           manifest_path, save_source_bundle, load_source_bundle,
                           source_bundle_path, variant_id)
from h2cmi.train.trainer import H2Model
from h2cmi.merge_grid_shards import merge_shards


def _tiny_cfg():
    c = core_config(H2Config(n_classes=3))
    c.encoder.n_chans = 8; c.encoder.n_times = 64
    return c


def test_config_signature_covers_full_config():
    a = _tiny_cfg()
    for mut in (lambda c: setattr(c.tta, "em_iters", 999),
                lambda c: setattr(c.tta, "trust_region", 7.0),
                lambda c: setattr(c.density, "cov_rank", 9),
                lambda c: setattr(c.train, "lr", 0.001234),
                lambda c: setattr(c.encoder, "n_chans", 99)):
        b = _tiny_cfg(); mut(b)
        assert config_signature(a) != config_signature(b), "signature missed a config field"
    assert config_signature(a) == config_signature(_tiny_cfg()), "signature not stable"


def test_variant_id_normalised():
    assert variant_id(resp="gen", update="oneshot", alpha=0.5, prior=None) == \
        "alpha=0.5__resp=gen__update=oneshot"


def test_manifest_guard():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "g.jsonl")
        m1 = build_manifest(_tiny_cfg(), ["cov"], "action", {"epochs": 1})
        validate_or_create_manifest(out, m1)                 # creates
        assert Path(manifest_path(out)).exists()
        validate_or_create_manifest(out, m1)                 # idempotent on match
        c2 = _tiny_cfg(); c2.tta.em_iters = 5
        m2 = build_manifest(c2, ["cov"], "action", {"epochs": 1})
        try:
            validate_or_create_manifest(out, m2)
            assert False, "config-mismatch manifest should abort"
        except RuntimeError:
            pass


def test_source_bundle_roundtrip_and_verification():
    cfg = _tiny_cfg()
    pi = np.full(3, 1 / 3)
    model = H2Model(cfg, pi)
    with tempfile.TemporaryDirectory() as d:
        p = source_bundle_path(d, 0, 1, "off")
        save_source_bundle(p, model, cfg, pi, source_data_hash="abc", commit_sha="deadbeef", history=[])
        m2, meta = load_source_bundle(p, build_model=lambda: H2Model(cfg, pi),
                                      expected_data_hash="abc",
                                      expected_config_signature=config_signature(cfg))
        # loaded weights identical to saved
        for (k, a), (_, b) in zip(model.state_dict().items(), m2.state_dict().items()):
            assert torch.allclose(a, b)
        # data-hash mismatch aborts
        try:
            load_source_bundle(p, build_model=lambda: H2Model(cfg, pi),
                               expected_data_hash="WRONG",
                               expected_config_signature=config_signature(cfg))
            assert False
        except RuntimeError:
            pass
        # config-signature mismatch aborts
        try:
            load_source_bundle(p, build_model=lambda: H2Model(cfg, pi),
                               expected_data_hash="abc", expected_config_signature="WRONG")
            assert False
        except RuntimeError:
            pass
        # corrupted stored checkpoint hash aborts
        b = torch.load(p, weights_only=False); b["source_checkpoint_hash"] = "0" * 12
        torch.save(b, p)
        try:
            load_source_bundle(p, build_model=lambda: H2Model(cfg, pi),
                               expected_data_hash="abc",
                               expected_config_signature=config_signature(cfg))
            assert False
        except RuntimeError:
            pass


def _shard(path, manifest, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(manifest_path(path), "w") as f:
        json.dump(manifest, f)


def _row(seed, site, action, cmi="off", scen="cov"):
    return dict(data_seed=seed, target_site=site, scenario=scen, action=action, cmi=cmi)


def test_shard_merge_strict():
    cfg = _tiny_cfg()
    man = build_manifest(cfg, ["cov"], "action", {})
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "shards"; sd.mkdir()
        _shard(str(sd / "a.jsonl"), man, [_row(0, 0, "identity"), _row(0, 0, "joint")])
        _shard(str(sd / "b.jsonl"), man, [_row(0, 1, "identity"), _row(0, 1, "joint")])
        out = str(Path(d) / "merged.jsonl")
        info = merge_shards(str(sd), out, item_field="action")
        assert info["rows"] == 4 and info["unique_keys"] == 4
        assert sum(1 for _ in open(out)) == 4
        # duplicate key across shards -> raise
        _shard(str(sd / "c.jsonl"), man, [_row(0, 0, "identity")])
        try:
            merge_shards(str(sd), out, item_field="action")
            assert False
        except ValueError:
            pass
        # differing run_signature -> raise
        Path(sd / "c.jsonl").unlink(); Path(manifest_path(str(sd / "c.jsonl"))).unlink()
        man2 = build_manifest(cfg, ["cov", "prior"], "action", {})
        _shard(str(sd / "d.jsonl"), man2, [_row(0, 2, "identity")])
        try:
            merge_shards(str(sd), out, item_field="action")
            assert False
        except RuntimeError:
            pass


if __name__ == "__main__":
    test_config_signature_covers_full_config()
    test_variant_id_normalised()
    test_manifest_guard()
    test_source_bundle_roundtrip_and_verification()
    test_shard_merge_strict()
    print("test_grid_io PASSED")
