"""Project B — real-EEG bridge (Step-3A).

Adapts the AAAI `cmi/` MOABB motor-imagery loader into the H2-CMI domain vocabulary so the Step-2
router harness can run on real EEG under a label-safe, source-only LOSO protocol. This is a pure
data/DAG adapter: it does NOT modify the MOABB loader, the H2-CMI trainer, the TTA, or the router.

First version is single-dataset LOSO with a two-factor subject->session DAG (no fabricated `site`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels


@dataclass(frozen=True)
class RealEEGDataset:
    X: np.ndarray
    y: np.ndarray
    meta: Any                      # pandas DataFrame with subject/session[/run]
    classes: list
    dataset: str
    fs: float
    subject_col: str = "subject"
    session_col: str = "session"


# ------------------------------------------------------------------ MOABB loader wrapper
def load_moabb_real_eeg(
    dataset: str,
    *,
    subjects: "Sequence[Any] | None" = None,
    max_subjects: "int | None" = None,
    tmin: float = 0.5,
    tmax: float = 3.5,
    resample: int = 128,
    binary: "bool | None" = None,
    normalize: str = "trial_zscore",
) -> RealEEGDataset:
    """Load a MOABB motor-imagery dataset via `cmi.data.moabb_data.load` (unmodified).

    If `subjects` is None and `max_subjects` is set, requests subjects 1..max_subjects (MOABB
    subject ids are 1-indexed ints); this bounds the download for a bridge smoke.
    """
    from cmi.data import moabb_data     # lazy: only needed when actually loading real data

    req = list(subjects) if subjects is not None else (
        list(range(1, int(max_subjects) + 1)) if max_subjects is not None else None)
    X, y, meta, classes = moabb_data.load(
        dataset, subjects=req, tmin=tmin, tmax=tmax, resample=resample,
        binary=binary, normalize=normalize)
    return RealEEGDataset(X=np.asarray(X, dtype=np.float32), y=np.asarray(y).astype(np.int64),
                          meta=meta.reset_index(drop=True), classes=list(classes),
                          dataset=str(dataset), fs=float(resample))


# ------------------------------------------------------------------ domain semantics
def make_subject_session_dag(meta, *, subject_col: str = "subject", session_col: str = "session"):
    """Build a subject->session H2-CMI DomainDAG + DomainLabels from real MOABB metadata.

    subject: random_effect, budget 0.05 (no parent).
    session: random_effect, budget 0.10, parent=subject; global level = unique subject|session pair.
    """
    subj = meta[subject_col].astype(str)
    pair = subj + "|" + meta[session_col].astype(str)
    subs = sorted(subj.unique())
    pairs = sorted(pair.unique())
    sub_to_level = {s: i for i, s in enumerate(subs)}
    sess_to_level = {p: i for i, p in enumerate(pairs)}

    subject = DomainFactor(name="subject", n_levels=len(subs), parents=(),
                           handling="random_effect", budget=0.05, description="MOABB subject")
    session = DomainFactor(name="session", n_levels=len(pairs), parents=("subject",),
                           handling="random_effect", budget=0.10, description="MOABB subject|session")
    dag = DomainDAG([subject, session])

    levels = np.stack([subj.map(sub_to_level).to_numpy(), pair.map(sess_to_level).to_numpy()], axis=1)
    labels = DomainLabels(dag, levels)
    info = dict(subject_to_level=sub_to_level, subject_session_to_level=sess_to_level,
                n_subjects=len(subs), n_sessions=len(pairs))
    return dag, labels, info


def make_source_domain_labels(meta_source, *, subject_col: str = "subject", session_col: str = "session"):
    """Source-side DAG + labels (thin wrapper over make_subject_session_dag)."""
    return make_subject_session_dag(meta_source, subject_col=subject_col, session_col=session_col)


def source_pseudo_levels_from_domains(domains: DomainLabels, *, level: str = "subject") -> np.ndarray:
    """Pseudo-target unit levels for source calibration (v1: source subjects)."""
    return domains.factor(level)


# ------------------------------------------------------------------ LOSO splits (label-safe)
def loso_subjects(meta, *, subject_col: str = "subject") -> list:
    return sorted(meta[subject_col].unique().tolist())


def split_loso_by_subject(meta, target_subject, *, subject_col: str = "subject"):
    """Return (source_idx, target_idx) index arrays for a leave-one-subject-out split."""
    tgt_mask = (meta[subject_col] == target_subject).to_numpy()
    return np.where(~tgt_mask)[0], np.where(tgt_mask)[0]


def target_domain_levels(meta_target, *, eval_unit: str = "subject", subject_col: str = "subject",
                         session_col: str = "session", run_col: str = "run") -> np.ndarray:
    """Per-trial evaluation-unit level for the target split (subject / session / run)."""
    subj = meta_target[subject_col].astype(str)
    if eval_unit == "subject":
        keys = subj
    elif eval_unit == "session":
        keys = subj + "|" + meta_target[session_col].astype(str)
    elif eval_unit == "run":
        if run_col not in meta_target.columns:
            raise ValueError(f"eval_unit='run' but column {run_col!r} not in meta {list(meta_target.columns)}")
        keys = subj + "|" + meta_target[session_col].astype(str) + "|" + meta_target[run_col].astype(str)
    else:
        raise ValueError(f"unknown eval_unit {eval_unit!r} (expected subject/session/run)")
    uniq = {k: i for i, k in enumerate(sorted(keys.unique()))}
    return keys.map(uniq).to_numpy()
