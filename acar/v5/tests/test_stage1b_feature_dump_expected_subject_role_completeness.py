"""Guard (Stage-1B8): the finalize barrier verifies the feature dump is COMPLETE + ref-consistent against an expected manifest — the
subject set must equal the expected fold subjects, each subject's split_role must match, and dump ref/disease/fold/seed must match the
ref. A missing subject, an extra subject, or a wrong role fails closed. Synthetic temp files only."""
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
ROLE = {f"{DISEASE}/ds/sub-1": "train", f"{DISEASE}/ds/sub-2": "val",
        f"{DISEASE}/ds/sub-3": "cal", f"{DISEASE}/ds/sub-4": "eval"}


def _write(dirpath, records):
    p = os.path.join(dirpath, "feat_dump.npz")
    FDW.write_feature_dump(p, ref=REF, disease=DISEASE, fold=FOLD, seed=SEED, records=records, **H)
    return {REF: {"feat_dump_path": p}}


def _records(role_map, nwin=1):
    return [(sk, role, w, [0.0, 1.0]) for sk, role in role_map.items() for w in range(nwin)]


def test_complete_dump_passes():
    with tempfile.TemporaryDirectory() as d:
        paths = _write(d, _records(ROLE))
        FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}})
    ok("a dump with exactly the expected subjects + matching roles → passes finalize completeness")


def test_missing_and_extra_subject_rejected():
    with tempfile.TemporaryDirectory() as d:
        missing = {k: v for k, v in ROLE.items() if k != f"{DISEASE}/ds/sub-4"}
        paths = _write(d, _records(missing))
        expect_raises(FIN.Stage1bFinalizeError, lambda: FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}}))
    with tempfile.TemporaryDirectory() as d:
        extra = dict(ROLE, **{f"{DISEASE}/ds/sub-9": "train"})
        paths = _write(d, _records(extra))
        expect_raises(FIN.Stage1bFinalizeError, lambda: FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}}))
    ok("a dump missing an expected subject, or carrying an extra subject → finalize completeness fails")


def test_wrong_role_rejected():
    with tempfile.TemporaryDirectory() as d:
        wrong = dict(ROLE, **{f"{DISEASE}/ds/sub-3": "train"})   # sub-3 expected 'cal', dumped as 'train'
        paths = _write(d, _records(wrong))
        expect_raises(FIN.Stage1bFinalizeError, lambda: FIN._validate_feature_dumps(paths, {REF: {"role_by_subject": ROLE}}))
    ok("a subject whose dumped split_role disagrees with the expected role → finalize completeness fails")


def test_ref_provenance_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "feat_dump.npz")
        FDW.write_feature_dump(p, ref=REF, disease=DISEASE, fold=FOLD, seed=SEED + 1, records=_records(ROLE), **H)  # wrong seed
        expect_raises(FIN.Stage1bFinalizeError,
                      lambda: FIN._validate_feature_dumps({REF: {"feat_dump_path": p}}, {REF: {"role_by_subject": ROLE}}))
    ok("a dump whose disease/fold/seed disagrees with the ref → finalize fails (provenance consistency)")


def main():
    print("ACAR v5 Stage-1B8 guard: feature dump expected subject/role completeness")
    test_complete_dump_passes()
    test_missing_and_extra_subject_rejected()
    test_wrong_role_rejected()
    test_ref_provenance_mismatch_rejected()
    print("ALL V5 STAGE1B-FEATURE-DUMP-COMPLETENESS GUARDS PASS")


if __name__ == "__main__":
    main()
