"""Guard (Stage-1B9): the launch guard requires output_root/run_id to be ABSENT or EMPTY — any existing content (a file, a
FINALIZED.json, a ref dir) aborts fail-closed. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_launch_guard as LG
from acar.v5.tests._util import expect_raises, ok

RUN = "run-syn-0001"


def test_absent_and_empty_are_fresh():
    with tempfile.TemporaryDirectory() as d:
        assert LG.assert_fresh_run_root(d, RUN) is True       # run root absent
        os.makedirs(os.path.join(d, RUN))                     # empty run root
        assert LG.assert_fresh_run_root(d, RUN) is True
    ok("absent run root, or an empty run root → fresh (allowed)")


def test_non_empty_run_root_rejected():
    for name in ("FINALIZED.json", "registry.json", "PD_fold0_seed20260711", "stray.bin"):
        with tempfile.TemporaryDirectory() as d:
            run_root = os.path.join(d, RUN)
            os.makedirs(run_root)
            target = os.path.join(run_root, name)
            if name.endswith((".json", ".bin")):
                open(target, "w").close()
            else:
                os.makedirs(target)                           # a per-ref dir from a prior run
            expect_raises(LG.Stage1bLaunchError, lambda d=d: LG.assert_fresh_run_root(d, RUN))
    ok("any existing content (FINALIZED.json / registry.json / ref dir / stray file) → Stage1bLaunchError")


def test_run_root_is_file_rejected():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, RUN), "w") as f:
            f.write("x")
        expect_raises(LG.Stage1bLaunchError, lambda: LG.assert_fresh_run_root(d, RUN))
    ok("run root exists but is a file (not a dir) → Stage1bLaunchError")


def test_symlinked_run_root_rejected():
    # a symlinked run root (even → an empty external dir) is rejected: realpath would collapse file+base and let artifacts escape
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as external:
        os.symlink(external, os.path.join(d, RUN))
        expect_raises(LG.Stage1bLaunchError, lambda: LG.assert_fresh_run_root(d, RUN))
    ok("a symlinked run root (pointing to an external dir) → Stage1bLaunchError (no artifact escape)")


def main():
    print("ACAR v5 Stage-1B9 guard: run root must be fresh")
    test_absent_and_empty_are_fresh()
    test_non_empty_run_root_rejected()
    test_run_root_is_file_rejected()
    test_symlinked_run_root_rejected()
    print("ALL V5 STAGE1B-RUN-ROOT-FRESH GUARDS PASS")


if __name__ == "__main__":
    main()
