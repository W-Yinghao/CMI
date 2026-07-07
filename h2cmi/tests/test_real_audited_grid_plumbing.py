"""Project A — grid-runner plumbing: sharding, resume, overwrite (synthetic, minimal training).

Uses --dataset synthetic --epochs 1 so each executed cell trains a tiny model quickly; the
sharding-coverage and resume tests are structured so most cells are NOT trained. Run:

    python -m h2cmi.tests.test_real_audited_grid_plumbing
"""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from h2cmi.run_real_audited_grid import main as grid_main  # noqa: E402


def _run(root, extra):
    argv = ["--dataset", "synthetic", "--n-classes", "2", "--epochs", "1", "--fast",
            "--align-factor", "site", "--device", "cpu", "--root-outdir", str(root)] + extra
    return grid_main(argv)


def test_grid_overwrite_rejects_resume_conflict():
    with tempfile.TemporaryDirectory() as root:
        raised = False
        try:
            _run(root, ["--target-subjects", "0", "--seeds", "0", "--resume", "--overwrite"])
        except SystemExit:
            raised = True                                 # argparse error() -> SystemExit
        assert raised, "--resume + --overwrite must be rejected"


def test_grid_sharding_covers_each_cell_once():
    # 2 cells (targets 0,1 x seed 0) split across 2 shards -> each shard runs exactly one cell
    with tempfile.TemporaryDirectory() as root:
        _run(root, ["--target-subjects", "0", "1", "--seeds", "0",
                    "--num-shards", "2", "--shard-index", "0"])
        _run(root, ["--target-subjects", "0", "1", "--seeds", "0",
                    "--num-shards", "2", "--shard-index", "1"])
        dirs = sorted(p.name for p in Path(root).glob("dataset=*"))
        assert dirs == ["dataset=synthetic_target=0_seed=0",
                        "dataset=synthetic_target=1_seed=0"], dirs
        for d in Path(root).glob("dataset=*"):
            assert json.loads((d / "run_manifest.json").read_text())["status"] == "ok"


def test_grid_resume_skips_completed_runs():
    with tempfile.TemporaryDirectory() as root:
        d = Path(root) / "dataset=synthetic_target=0_seed=0"
        d.mkdir(parents=True)
        sentinel = {"status": "ok", "dataset": "synthetic", "target_subject": 0, "seed": 0,
                    "_sentinel": True}
        (d / "run_manifest.json").write_text(json.dumps(sentinel))
        _run(root, ["--target-subjects", "0", "--seeds", "0", "--resume"])
        # resume must NOT re-run / overwrite the already-complete cell
        assert json.loads((d / "run_manifest.json").read_text()).get("_sentinel") is True


ALL_TESTS = [test_grid_overwrite_rejects_resume_conflict,
             test_grid_sharding_covers_each_cell_once,
             test_grid_resume_skips_completed_runs]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} GRID-PLUMBING TESTS PASSED")


if __name__ == "__main__":
    run()
