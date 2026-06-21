"""Compute-resume integration: a complete resume must NOT retrain; a partial resume must
REUSE the exact source bundle (not retrain); a config change must ABORT, not append."""
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

BASE = ["run_action_grid", "--scenarios", "population_null,cov", "--target-sites", "all",
        "--sites", "2", "--subjects", "2", "--sessions", "1", "--trials", "8", "--fast"]


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
        _run(["--seeds", "0", "--epochs", "1", "--out", out])
        n1 = sum(1 for _ in open(out))
        orig = run_action_grid.train_h2
        run_action_grid.train_h2 = _forbid_train          # any train call now fails
        try:
            _run(["--seeds", "0", "--epochs", "1", "--out", out])   # all keys done -> skip train
        finally:
            run_action_grid.train_h2 = orig
        assert sum(1 for _ in open(out)) == n1


def test_partial_resume_reuses_bundle_not_train():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "ag.jsonl")
        _run(["--seeds", "0", "--epochs", "1", "--out", out])
        rows = [json.loads(l) for l in open(out)]
        n_full = len(rows)
        # drop every 'joint' row -> those keys are no longer done (bundles still on disk)
        kept = [r for r in rows if r["action"] != "joint"]
        assert len(kept) < n_full
        with open(out, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        orig = run_action_grid.train_h2
        run_action_grid.train_h2 = _forbid_train          # must LOAD the bundle, not retrain
        try:
            _run(["--seeds", "0", "--epochs", "1", "--out", out])
        finally:
            run_action_grid.train_h2 = orig
        final = [json.loads(l) for l in open(out)]
        assert len(final) == n_full, "partial resume did not refill the missing rows"
        keys = [(r["data_seed"], r["target_site"], r["scenario"], r["action"], r["cmi"]) for r in final]
        assert len(keys) == len(set(keys)), "duplicate rows after partial resume"


def test_resume_rejects_config_mismatch():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "ag.jsonl")
        _run(["--seeds", "0", "--epochs", "1", "--out", out])
        try:
            _run(["--seeds", "0", "--epochs", "2", "--out", out])   # different config_signature
            assert False, "config-mismatch resume should abort"
        except RuntimeError:
            pass


if __name__ == "__main__":
    test_complete_resume_skips_train_h2()
    test_partial_resume_reuses_bundle_not_train()
    test_resume_rejects_config_mismatch()
    print("test_compute_resume PASSED")
