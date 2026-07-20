"""BNCI2014_001 leave-one-subject-out (LOSO) fold plan -- deterministic cyclic split (C6).

For each held-out target subject ``t`` (1..9), the remaining subjects are taken in cyclic order starting
just after ``t``: the next two are the source-audit subjects, the other six are source-train, and the
level-1 deleted cell is the first source-train subject (in the same cyclic order) crossed with ``feet``.
This holds each subject out exactly once and rotates the audit / deleted roles deterministically. The
target-001 fold reproduces the C5 split (audit 002,003 / train 004..009 / deleted 004).
"""
from __future__ import annotations

SUBJECTS = tuple(range(1, 10))                 # BNCI2014_001 (BCI IV-2a) has nine subjects
DELETED_CLASS = "feet"


def cyclic_after(target, subjects=SUBJECTS):
    """The subjects after ``target`` in cyclic order (length len(subjects)-1)."""
    n = len(subjects)
    i = subjects.index(int(target))
    return [subjects[(i + k) % n] for k in range(1, n)]


def loso_fold_spec(target, *, subjects=SUBJECTS, source_audit_count=2, deleted_class=DELETED_CLASS,
                   dataset_id="BNCI2014_001") -> dict:
    after = cyclic_after(target, subjects)
    audit = after[:source_audit_count]
    train = after[source_audit_count:]
    return {"fold_id": f"target-{int(target):03d}", "dataset_id": dataset_id, "target": int(target),
            "subjects": sorted(int(s) for s in subjects),
            "source_audit_subjects": list(audit), "source_train_subjects": list(train),
            "deleted_subject": int(train[0]), "deleted_class": str(deleted_class),
            "deleted_cell": {"domain_id": f"{dataset_id}|subject-{int(train[0]):03d}", "class_name": str(deleted_class)}}


def loso_plan(subjects=SUBJECTS, **kw) -> list:
    """The nine LOSO folds (one per held-out target), each a disjoint 1/2/6 subject split."""
    return [loso_fold_spec(t, subjects=subjects, **kw) for t in subjects]


def explicit_split(spec) -> dict:
    """The materialize.py override payload (subjects + role lists) for a LOSO fold spec."""
    return {"subjects": list(spec["subjects"]), "target_subjects": [int(spec["target"])],
            "source_audit_subjects": list(spec["source_audit_subjects"]),
            "source_train_subjects": list(spec["source_train_subjects"])}
