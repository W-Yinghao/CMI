"""ACAR V5 Stage-1B subject eligibility (pure/stdlib). Decides the Stage-1B subject universe BEFORE the split, so a subject cannot
enter CAL/EVAL and only fail later when a label is needed. Eligibility requires, for every subject in the disease index: a raw
recording location (the subject came from list_subjects, so it exists) AND a resolvable control/case label. Resolvability is checked
via the reader's `subject_label_resolvable(...)` which returns a BOOLEAN ONLY — the label value never leaves the reader, so this is
not a training-label read: the label-free embedding view still has no label path, and the FIT view remains the only place a label
VALUE is read (for training). Fail-closed: any ineligible subject aborts the build before any split/train/dump.
"""
from __future__ import annotations


class SubjectEligibilityError(RuntimeError):
    pass


def assert_all_eligible(index, dev_reader, cohort_paths):
    """Verify every subject in `index` is eligible (raw location + resolvable label). Fail-closed with the ineligible list."""
    resolver = getattr(dev_reader, "subject_label_resolvable", None)
    if not callable(resolver):
        raise SubjectEligibilityError("reader is missing subject_label_resolvable() — cannot verify subject eligibility")
    ineligible = []
    for key in index.subject_keys:
        cohort, raw = index.cohort_of(key), index.raw_of(key)
        path = cohort_paths.get(cohort)
        if path is None:
            ineligible.append((key, "no_cohort_path"))
            continue
        try:
            ok = bool(resolver(index.disease, cohort, raw, path))   # BOOLEAN only — the label value is never returned here
        except Exception as ex:  # noqa: BLE001 — any resolver failure is ineligibility, fail-closed
            ineligible.append((key, f"resolver_error:{type(ex).__name__}"))
            continue
        if not ok:
            ineligible.append((key, "label_unresolvable"))
    if ineligible:
        raise SubjectEligibilityError(f"{len(ineligible)} ineligible subject(s) before split (fail-closed): {ineligible[:5]}")
    return True
