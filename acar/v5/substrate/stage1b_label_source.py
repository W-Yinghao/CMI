"""ACAR V5 Stage-1B label source contract (pure/stdlib). Labels are read ONLY for FIT subjects and ONLY via
`AuthorizedFitDatasetView.read_label`; the label-free embedding-dump view has no label path at all. This module pins the
label MAPPING and a fail-closed resolver so the real label read (BIDS participants.tsv / diagnosis field, wired at the real run)
can only ever produce the two pinned integer classes; anything unknown / missing / ambiguous fails closed. No I/O here (the raw
metadata read is the reader's seam); this is the mapping + validation the reader MUST apply before returning a label.
"""
from __future__ import annotations

# pinned binary mapping (matches training_config.label_mapping); accepted spellings normalize to a canonical token
LABEL_MAPPING = {"control": 0, "case": 1}

_CONTROL_ALIASES = ("control", "hc", "healthy", "healthy_control", "hc_control", "0")
_CASE_ALIASES = ("case", "patient", "pd", "scz", "disease", "1")


class LabelSourceError(RuntimeError):
    """Raised when a raw label value is unknown / missing / ambiguous — fail-closed (never silently defaulted)."""


def normalize_group(raw_value):
    """Map a raw participant group string to the canonical token 'control'/'case'. Fail-closed: missing/empty/unknown → raise."""
    if raw_value is None:
        raise LabelSourceError("missing group label (None) — fail-closed")
    if not isinstance(raw_value, str):
        raise LabelSourceError(f"group label must be a string, got {type(raw_value).__name__}")
    tok = raw_value.strip().lower().replace("-", "_")
    if not tok:
        raise LabelSourceError("empty group label — fail-closed")
    is_ctrl, is_case = tok in _CONTROL_ALIASES, tok in _CASE_ALIASES
    if is_ctrl and is_case:
        raise LabelSourceError(f"ambiguous group label {raw_value!r}")
    if is_ctrl:
        return "control"
    if is_case:
        return "case"
    raise LabelSourceError(f"unknown group label {raw_value!r} (not a recognized control/case value) — fail-closed")


def resolve_label(raw_value):
    """Raw group value → pinned integer class {0,1}. Fail-closed on anything not clearly control/case."""
    return LABEL_MAPPING[normalize_group(raw_value)]


# ---- BIDS participants.tsv reader (pure stdlib; the label-source seam the FIT reader delegates to) --------------------------
GROUP_COLUMNS = ("group", "diagnosis", "dx", "participant_group")   # first present column, in this order


def read_participant_group(participants_tsv_path, participant_id):
    """Deterministic participants.tsv read: return the raw group cell for `participant_id`. Fail-closed on missing file / missing
    subject / no group column / duplicate subject rows. Pure stdlib (csv), no pandas, no heavy deps."""
    import csv  # stdlib, lazy
    import os
    if not participants_tsv_path or not os.path.isfile(participants_tsv_path):
        raise LabelSourceError(f"participants.tsv not found: {participants_tsv_path}")
    with open(participants_tsv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise LabelSourceError("participants.tsv has no rows")
    header = rows[0].keys()
    group_col = next((c for c in GROUP_COLUMNS if c in header), None)
    if group_col is None:
        raise LabelSourceError(f"participants.tsv has no group column (looked for {GROUP_COLUMNS})")
    matches = [r for r in rows if r.get("participant_id") == participant_id]
    if len(matches) == 0:
        raise LabelSourceError(f"participant {participant_id!r} not found in participants.tsv")
    if len(matches) > 1:
        raise LabelSourceError(f"participant {participant_id!r} has {len(matches)} rows (ambiguous)")
    return matches[0].get(group_col)


def resolve_subject_label(participants_tsv_path, participant_id):
    """participants.tsv + subject id → pinned integer class {0,1}. Fail-closed at every step."""
    return resolve_label(read_participant_group(participants_tsv_path, participant_id))
