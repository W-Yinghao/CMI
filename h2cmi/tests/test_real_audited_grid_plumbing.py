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


def test_grid_manifest_expected_cells_complete():
    with tempfile.TemporaryDirectory() as root:
        _run(root, ["--target-subjects", "0", "1", "--seeds", "0"])
        gm = json.loads((Path(root) / "grid_manifest.json").read_text())
        cells = {(c["target_subject"], c["seed"]) for c in gm["expected_cells"]}
        assert cells == {(0, 0), (1, 0)} and gm["n_classes"] == 2
        from h2cmi.observability.validate_results import main as vmain
        assert vmain(["--root", root]) == 0            # reads grid_manifest -> complete + valid


def test_grid_manifest_missing_cell_fails_by_default():
    with tempfile.TemporaryDirectory() as root:
        # run only shard 0 of 2 -> only 1 of the 2 grid_manifest cells materializes
        _run(root, ["--target-subjects", "0", "1", "--seeds", "0",
                    "--num-shards", "2", "--shard-index", "0"])
        from h2cmi.observability.validate_results import main as vmain
        assert vmain(["--root", root]) == 1            # missing cell fails by default
        assert vmain(["--root", root, "--allow-missing"]) == 0


def test_grid_manifest_legal_skip_counts_as_present():
    with tempfile.TemporaryDirectory() as root:
        # target 999 has no trials -> legal skip artifact; it must count as PRESENT (not missing)
        _run(root, ["--target-subjects", "0", "999", "--seeds", "0"])
        d999 = Path(root) / "dataset=synthetic_target=999_seed=0"
        m = json.loads((d999 / "run_manifest.json").read_text())
        assert m["status"] == "skipped" and m.get("skip_reason") and m.get("seed") == 0
        from h2cmi.observability.validate_results import main as vmain
        assert vmain(["--root", root]) == 0            # skip present -> no missing cell


def test_grid_all_targets_resolved_after_load_with_fake_subject_map():
    # 'all' on a (faked) MOABB dataset resolves target ids from the subject_map, post-load
    import h2cmi.run_real_audited as R
    orig = R._load_moabb

    def fake_load(name, subjects, seed):
        X, y, dag, domains, subj_col, sess, _ncls, info = R._load_synthetic(2, 0)
        info = dict(info)
        info["subject_map"] = {"3": 0, "5": 1, "7": 2}      # non-contiguous MOABB subject ids
        return X, y, dag, domains, subj_col, sess, 2, info

    R._load_moabb = fake_load
    try:
        with tempfile.TemporaryDirectory() as root:
            grid_main(["--dataset", "FakeMOABB", "--target-subjects", "all", "--seeds", "0",
                       "--epochs", "1", "--fast", "--align-factor", "site", "--device", "cpu",
                       "--num-shards", "3", "--shard-index", "0", "--root-outdir", root])
            gm = json.loads((Path(root) / "grid_manifest.json").read_text())
            assert gm["target_subjects"] == [3, 5, 7]        # resolved sorted int keys, not 'all'
            assert gm["subjects"] is None                    # 'all' load -> subjects None
            assert len(gm["expected_cells"]) == 3
    finally:
        R._load_moabb = orig


def test_grid_manifest_records_resolved_targets_not_literal_all():
    with tempfile.TemporaryDirectory() as root:
        _run(root, ["--target-subjects", "all", "--seeds", "0",
                    "--num-shards", "6", "--shard-index", "0"])
        gm = json.loads((Path(root) / "grid_manifest.json").read_text())
        assert "all" not in gm["target_subjects"]            # concrete ids, never the literal 'all'
        assert gm["target_subjects"] == sorted(gm["target_subjects"])
        assert gm["target_subjects"] and all(isinstance(t, int) for t in gm["target_subjects"])


ALL_TESTS = [test_grid_overwrite_rejects_resume_conflict,
             test_grid_sharding_covers_each_cell_once,
             test_grid_resume_skips_completed_runs,
             test_grid_manifest_expected_cells_complete,
             test_grid_manifest_missing_cell_fails_by_default,
             test_grid_manifest_legal_skip_counts_as_present,
             test_grid_all_targets_resolved_after_load_with_fake_subject_map,
             test_grid_manifest_records_resolved_targets_not_literal_all]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} GRID-PLUMBING TESTS PASSED")


if __name__ == "__main__":
    run()
