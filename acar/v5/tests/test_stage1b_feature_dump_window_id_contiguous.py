"""Guard (Stage-1B8): finalize requires each subject's window_ids in the feature dump to be EXACTLY 0..n-1 (contiguous, unique) —
so no window is dropped, duplicated, or mis-indexed. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_finalize as FIN
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import expect_raises, ok

REF = sorted(SA.CANONICAL_FOLD_REFS)[0]
DISEASE = REF.split("/")[0]
FOLD = int(REF.split("fold")[1].split("/")[0])
SEED = int(REF.split("seed")[1])
H = dict(preprocessing_config_sha256="a" * 64, training_config_sha256="b" * 64,
         encoder_checkpoint_file_sha256="c" * 64, source_state_file_sha256="d" * 64)
SUB = f"{DISEASE}/ds/sub-1"
ROLE = {SUB: "train"}


def _write(d, window_ids):
    p = os.path.join(d, "feat_dump.npz")
    FDW.write_feature_dump(p, ref=REF, disease=DISEASE, fold=FOLD, seed=SEED,
                           records=[(SUB, "train", w, [0.0, 1.0]) for w in window_ids], **H)
    return {REF: {"feat_dump_path": p}}


def test_contiguous_window_ids_pass():
    with tempfile.TemporaryDirectory() as d:
        paths = _write(d, [0, 1, 2])
        FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}})
    ok("window_ids 0,1,2 for a subject → contiguous → passes")


def test_gapped_and_duplicated_window_ids_rejected():
    with tempfile.TemporaryDirectory() as d:
        paths = _write(d, [0, 2])                             # gap (missing window 1)
        expect_raises(FIN.Stage1bFinalizeError, lambda: FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}}))
    with tempfile.TemporaryDirectory() as d:
        paths = _write(d, [0, 0])                             # duplicate window id
        expect_raises(FIN.Stage1bFinalizeError, lambda: FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}}))
    ok("gapped (0,2) or duplicated (0,0) window_ids → finalize fails (must be exactly 0..n-1)")


def test_window_count_mismatch_rejected():
    # a contiguous-but-TOO-FEW dump: window_ids [0] passes contiguity, but expected n_windows=3 → count mismatch fails
    with tempfile.TemporaryDirectory() as d:
        paths = _write(d, [0])
        exp = {REF: {"role_by_subject": ROLE, "n_windows_by_subject": {SUB: 3}}}
        expect_raises(FIN.Stage1bFinalizeError, lambda: FIN._validate_feature_dumps(paths, exp))
    ok("a subject with fewer windows than the authoritative expected count → finalize fails (count, not just contiguity)")


def main():
    print("ACAR v5 Stage-1B8 guard: feature dump window_id contiguous")
    test_contiguous_window_ids_pass()
    test_gapped_and_duplicated_window_ids_rejected()
    test_window_count_mismatch_rejected()
    print("ALL V5 STAGE1B-FEATURE-DUMP-WINDOW-ID GUARDS PASS")


if __name__ == "__main__":
    main()
