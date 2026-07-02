"""ACAR V5 Stage-1B cohort-specific label spec (pure/stdlib; csv is lazy). Each frozen DEV cohort encodes the control/case diagnosis
differently (a participants.tsv column with cohort-specific header + values, or the subject-id prefix). Rather than a broad global
alias set, a FROZEN per-cohort table pins EXACTLY how each of the 7 cohorts' labels are read. Fail-closed: unknown cohort / missing
column / missing subject / unrecognized value all raise. The value never leaks — the eligibility resolver returns a boolean; the
label VALUE is only produced through AuthorizedFitDatasetView.read_label (FIT training only); the embedding view has no label path.
"""
from __future__ import annotations
from acar.v5.substrate.stage1b_label_source import LABEL_MAPPING   # {"control": 0, "case": 1}

# (disease, cohort) -> how to read the label. mode="column": a participants.tsv column (matched case-insensitively) with the two
# pinned values; mode="id_prefix": the subject-id prefix (after stripping "sub-").
COHORT_LABEL_SPEC = {
    ("PD", "ds002778"): {"mode": "id_prefix", "control": "hc", "case": "pd"},
    ("PD", "ds003490"): {"mode": "column", "column": "Group", "control": "CTL", "case": "PD"},
    ("PD", "ds004584"): {"mode": "column", "column": "GROUP", "control": "Control", "case": "PD"},
    ("SCZ", "ds003944"): {"mode": "column", "column": "type", "control": "Control", "case": "Psychosis"},
    ("SCZ", "ds003947"): {"mode": "column", "column": "type", "control": "Control", "case": "Psychosis"},
    ("SCZ", "ds004000"): {"mode": "column", "column": "group", "control": "HC", "case": "P"},
    ("SCZ", "ds004367"): {"mode": "column", "column": "Group", "control": "Control", "case": "Patient"},
}


class CohortLabelSpecError(RuntimeError):
    pass


def _norm(s):
    return str(s).strip().casefold()


def _match_column(header, requested):
    """Case-insensitive, whitespace-stripped column match. If multiple columns collapse to the requested name → FAIL."""
    req = _norm(requested)
    matches = [c for c in header if _norm(c) == req]
    if not matches:
        raise CohortLabelSpecError(f"column {requested!r} not found in {list(header)}")
    if len(matches) > 1:
        raise CohortLabelSpecError(f"multiple columns collapse to {requested!r}: {matches}")
    return matches[0]


def resolve_label(disease, cohort, subject, participants_tsv_path):
    """Cohort-exact label resolution → pinned integer class {0,1}. Fail-closed at every step (no disease-wide fallback, no path
    inference)."""
    spec = COHORT_LABEL_SPEC.get((disease, cohort))
    if spec is None:
        raise CohortLabelSpecError(f"no label spec for cohort {disease}/{cohort}")
    ctrl, case = _norm(spec["control"]), _norm(spec["case"])
    if spec["mode"] == "id_prefix":
        raw = str(subject)
        if raw.startswith("sub-"):
            raw = raw[len("sub-"):]
        raw = _norm(raw)
        is_ctrl, is_case = raw.startswith(ctrl), raw.startswith(case)
        if is_ctrl and is_case:
            raise CohortLabelSpecError(f"{disease}/{cohort} {subject!r}: id prefix is ambiguous (control={ctrl}/case={case})")
        if is_ctrl:
            return LABEL_MAPPING["control"]
        if is_case:
            return LABEL_MAPPING["case"]
        raise CohortLabelSpecError(f"{disease}/{cohort} {subject!r}: id prefix matches neither control({spec['control']}) "
                                   f"nor case({spec['case']})")
    # column mode
    import csv
    import os
    if not participants_tsv_path or not os.path.isfile(participants_tsv_path):
        raise CohortLabelSpecError(f"participants.tsv not found: {participants_tsv_path}")
    with open(participants_tsv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise CohortLabelSpecError("participants.tsv has no rows")
    col = _match_column(rows[0].keys(), spec["column"])
    matches = [r for r in rows if r.get("participant_id") == subject]
    if len(matches) == 0:
        raise CohortLabelSpecError(f"participant {subject!r} not found in participants.tsv")
    if len(matches) > 1:
        raise CohortLabelSpecError(f"participant {subject!r} has {len(matches)} rows (ambiguous)")
    val = _norm(matches[0].get(col))
    if val == ctrl:
        return LABEL_MAPPING["control"]
    if val == case:
        return LABEL_MAPPING["case"]
    raise CohortLabelSpecError(f"{disease}/{cohort} {subject}: value {matches[0].get(col)!r} is neither "
                               f"control({spec['control']}) nor case({spec['case']})")


def label_resolvable(disease, cohort, subject, participants_tsv_path):
    """Boolean-only eligibility check — the label VALUE never leaves this function."""
    try:
        resolve_label(disease, cohort, subject, participants_tsv_path)
        return True
    except CohortLabelSpecError:
        return False
