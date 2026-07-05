"""ACAR V5 Stage-2 LABEL loader + firewall (pure/stdlib; numpy-free). Stage-2 is the first label-consuming V5 step, so the label
VALUE is confined behind a closure-backed, evaluation-only view: it can be resolved ONLY for a pre-authorized subject set, one
subject at a time, and NEVER in bulk. Labels enter only gate/metric code (policy evaluation); they are structurally unreachable
from routing / scalarization / threshold fitting (those never receive a label view).

Reuses the frozen cohort label spec (acar/v5/substrate/cohort_label_spec.py) — the exact 7-cohort spec, fail-closed — never the
looser alias resolver. A feat_dump canonical subject_key "disease/cohort/raw" is split and resolved via participants.tsv.
"""
from __future__ import annotations
from acar.v5.substrate import cohort_label_spec as CLS


class Stage2LabelError(RuntimeError):
    """Raised on a malformed subject_key, an unauthorized label request, or a cohort-label-spec failure (fail-closed)."""


def split_subject_key(subject_key):
    """Canonical subject_key 'disease/cohort/raw' -> (disease, cohort, raw). Fail-closed on any other shape."""
    parts = str(subject_key).split("/")
    if len(parts) != 3 or not all(parts):
        raise Stage2LabelError(f"bad canonical subject_key {subject_key!r} (want 'disease/cohort/raw')")
    return parts[0], parts[1], parts[2]


class EvaluationLabelView:
    """Closure-backed, evaluation-ONLY label view. Exposes ONLY `resolve_label(subject_key) -> {0,1}` for subjects in the
    authorized set. Holds the participants.tsv paths + authorized set in a closure — no attribute exposes them, and there is no
    bulk-dump method. This is the ONLY seam through which a Stage-2 label VALUE is produced (mirrors AuthorizedFitDatasetView)."""

    def __init__(self, participants_tsv_by_cohort, authorized_subject_keys):
        _paths = dict(participants_tsv_by_cohort)
        _allowed = frozenset(authorized_subject_keys)

        def _resolve(subject_key):
            if subject_key not in _allowed:
                raise Stage2LabelError(f"subject {subject_key!r} is not in the authorized evaluation set")
            disease, cohort, raw = split_subject_key(subject_key)
            if cohort not in _paths:
                raise Stage2LabelError(f"no participants.tsv path for cohort {cohort!r}")
            return int(CLS.resolve_label(disease, cohort, raw, _paths[cohort]))

        self._resolve = _resolve                                # the ONLY reference to labels/paths (closure-captured)

    def resolve_label(self, subject_key):
        return self._resolve(subject_key)


def make_evaluation_label_view(participants_tsv_by_cohort, authorized_subject_keys):
    """Build an EvaluationLabelView for the authorized (CAL∪EVAL) subject set. `participants_tsv_by_cohort` maps cohort ->
    participants.tsv path (None allowed for id_prefix cohorts like ds002778)."""
    return EvaluationLabelView(participants_tsv_by_cohort, authorized_subject_keys)


def label_resolvable(subject_key, participants_tsv_by_cohort):
    """Boolean-only eligibility (the label value never leaves this function)."""
    disease, cohort, raw = split_subject_key(subject_key)
    return bool(CLS.label_resolvable(disease, cohort, raw, dict(participants_tsv_by_cohort).get(cohort)))
