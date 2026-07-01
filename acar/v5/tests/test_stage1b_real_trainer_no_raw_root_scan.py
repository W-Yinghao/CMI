"""Guard (Stage-1B4): the real trainer reads signal ONLY via dataset_view.read_windows — it never scans the filesystem nor calls
reader methods directly. Source-level (no execution). Synthetic only."""
from __future__ import annotations
import inspect
from acar.v5.substrate import real_trainer as RT
from acar.v5.tests._util import ok

FORBIDDEN_SCAN = ("os.walk", "os.listdir", "os.scandir", "glob(", ".glob(", ".rglob(", "iterdir(", "os.path.join")
FORBIDDEN_DIRECT_READER = (".read_subject_windows(", ".list_subjects(")


def test_no_filesystem_scan_in_real_trainer():
    src = inspect.getsource(RT)
    for tok in FORBIDDEN_SCAN:
        assert tok not in src, f"real trainer must not scan the filesystem ({tok!r})"
    ok("real trainer source performs no filesystem scan (no os.walk/listdir/scandir/glob/rglob/join)")


def test_no_direct_reader_calls():
    src = inspect.getsource(RT)
    for tok in FORBIDDEN_DIRECT_READER:
        assert tok not in src, f"real trainer must not call the reader directly ({tok!r})"
    assert "dataset_view.read_windows(" in src, "real trainer must read via dataset_view.read_windows"
    ok("real trainer accesses signal ONLY via dataset_view.read_windows (no direct reader calls)")


def test_train_fold_signature():
    params = list(inspect.signature(RT.RealSubstrateTrainer.train_fold).parameters)
    assert params == ["self", "disease", "fold", "seed", "train_subject_keys", "val_subject_keys", "dataset_view"], params
    ok("real trainer.train_fold takes FIT subject keys + dataset_view (no cohort_paths / raw roots)")


def main():
    print("ACAR v5 Stage-1B4 guard: real trainer no raw root scan")
    test_no_filesystem_scan_in_real_trainer()
    test_no_direct_reader_calls()
    test_train_fold_signature()
    print("ALL V5 STAGE1B-REAL-TRAINER-NO-SCAN GUARDS PASS")


if __name__ == "__main__":
    main()
