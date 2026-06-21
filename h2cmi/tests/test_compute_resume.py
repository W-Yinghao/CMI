"""Compute-resume + real-shard integration: a complete resume must NOT retrain; a partial
resume must REUSE the verified source bundle; a config change must ABORT; and two real shard
invocations must share an experiment_signature and merge to the exact global key set."""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import pytest
    pytestmark = pytest.mark.integration
except ImportError:
    pass

from h2cmi import run_action_grid
from h2cmi.grid_io import manifest_path
from h2cmi.merge_grid_shards import merge_shards

BASE = ["run_action_grid", "--scenarios", "population_null,cov", "--grid-target-sites", "all",
        "--sites", "2", "--subjects", "2", "--sessions", "1", "--trials", "8", "--fast",
        "--allow-dirty"]


def _run(extra):
    old = sys.argv
    sys.argv = BASE + extra
    try:
        run_action_grid.main()
    finally:
        sys.argv = old


def _forbid_train(*a, **k):
    raise AssertionError("train_h2 must not run on resume")


def test_complete_resume_skips_train_h2():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "ag.jsonl")
        _run(["--grid-seeds", "0", "--epochs", "1", "--out", out])
        n1 = sum(1 for _ in open(out))
        orig = run_action_grid.train_h2
        run_action_grid.train_h2 = _forbid_train
        try:
            _run(["--grid-seeds", "0", "--epochs", "1", "--out", out])
        finally:
            run_action_grid.train_h2 = orig
        assert sum(1 for _ in open(out)) == n1


def test_partial_resume_reuses_bundle_not_train():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "ag.jsonl")
        _run(["--grid-seeds", "0", "--epochs", "1", "--out", out])
        rows = [json.loads(l) for l in open(out)]
        n_full = len(rows)
        kept = [r for r in rows if r["action"] != "joint"]       # drop a subset -> keys undone
        assert len(kept) < n_full
        with open(out, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        orig = run_action_grid.train_h2
        run_action_grid.train_h2 = _forbid_train                 # must LOAD bundle, not retrain
        try:
            _run(["--grid-seeds", "0", "--epochs", "1", "--out", out])
        finally:
            run_action_grid.train_h2 = orig
        final = [json.loads(l) for l in open(out)]
        assert len(final) == n_full
        keys = [(r["data_seed"], r["target_site"], r["scenario"], r["action"], r["cmi"]) for r in final]
        assert len(keys) == len(set(keys))


def test_resume_rejects_config_mismatch():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "ag.jsonl")
        _run(["--grid-seeds", "0", "--epochs", "1", "--out", out])
        try:
            _run(["--grid-seeds", "0", "--epochs", "2", "--out", out])   # different experiment sig
            assert False, "config-mismatch resume should abort"
        except RuntimeError:
            pass


def test_real_shard_invocations_share_signature_and_merge():
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "shards"; sd.mkdir()
        br = str(Path(d) / "bundles")
        common = ["--grid-seeds", "0", "--epochs", "1", "--bundle-root", br]
        _run(common + ["--shard-target-sites", "0", "--out", str(sd / "s0.jsonl")])
        _run(common + ["--shard-target-sites", "1", "--out", str(sd / "s1.jsonl")])
        m0 = json.load(open(manifest_path(str(sd / "s0.jsonl"))))
        m1 = json.load(open(manifest_path(str(sd / "s1.jsonl"))))
        assert m0["experiment_signature"] == m1["experiment_signature"]
        assert m0["shard_spec"] != m1["shard_spec"]
        out = str(Path(d) / "merged.jsonl")
        info = merge_shards(str(sd), out)                         # exact-key merge succeeds
        # 1 seed x 2 sites x 2 scenarios x 4 actions x 2 cmi = 32
        assert info["rows"] == 32 and info["unique_keys"] == 32


if __name__ == "__main__":
    test_complete_resume_skips_train_h2()
    test_partial_resume_reuses_bundle_not_train()
    test_resume_rejects_config_mismatch()
    test_real_shard_invocations_share_signature_and_merge()
    print("test_compute_resume PASSED")
