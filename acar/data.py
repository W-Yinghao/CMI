"""Data layer: load erm_0 dumps, fit serialized source state, build natural deployment batches.

A "cohort" is a held-out pseudo-target. Per cohort we fit the deployed source readout on (z_ev, y_ev) and treat
z_te as the unlabeled deployment stream. Batches are NATURAL: window-ordered, grouped by recording (= session),
chunked to B. y_te is carried alongside but is consumed ONLY by risk.py (Phase-2). Nothing here aggregates by label.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from cmi.eval.source_state import fit_source_state
from .config import feat_dump_dir, DISEASE, N_CLS, RHO, B, MIN_BATCH


@dataclass
class Batch:
    cohort: str
    disease: str
    recording: str
    z: np.ndarray            # [n, d] unlabeled target features (deployment input)
    y: np.ndarray            # [n] TRUE labels — Phase-2 only; never touched by scoring
    fallback: bool           # label-blind: forced identity (n < MIN_BATCH)


@dataclass
class Cohort:
    cohort: str
    disease: str
    state: dict              # serialized source state (frozen probe + moments); no raw source
    batches: list            # list[Batch]


def _load_npz(disease: str, cohort: str):
    o = np.load(f"{feat_dump_dir()}/audit_{disease}_{cohort}_erm_0.npz", allow_pickle=True)
    zev, yev = np.asarray(o["z_ev"], float), np.asarray(o["y_ev"]).astype(int)
    zte, yte = np.asarray(o["z_te"], float), np.asarray(o["y_te"]).astype(int)
    rec = np.asarray(o["recording_id_te"]).astype(str)
    win = np.asarray(o["window_index_te"]).astype(int)
    return zev, yev, zte, yte, rec, win


def _natural_batches(disease, cohort, zte, yte, rec, win, batch_size=B):
    """Window-ordered, recording-grouped, chunked to batch_size. Deterministic; label-blind chunking."""
    out = []
    for r in sorted(set(rec.tolist())):                       # stable recording order
        idx = np.where(rec == r)[0]
        idx = idx[np.argsort(win[idx], kind="stable")]        # natural acquisition order within a recording
        for s in range(0, len(idx), batch_size):
            sl = idx[s:s + batch_size]
            out.append(Batch(cohort=cohort, disease=disease, recording=r,
                             z=zte[sl], y=yte[sl], fallback=len(sl) < MIN_BATCH))
    return out


def load_cohort(disease: str, cohort: str, batch_size=B) -> Cohort:
    zev, yev, zte, yte, rec, win = _load_npz(disease, cohort)
    state = fit_source_state(zev, yev, N_CLS, rho=RHO)
    batches = _natural_batches(disease, cohort, zte, yte, rec, win, batch_size)
    return Cohort(cohort=cohort, disease=disease, state=state, batches=batches)


def load_all(batch_size=B) -> dict:
    """{disease: [Cohort, ...]} for all frozen cohorts."""
    return {d: [load_cohort(d, c, batch_size) for c in cohs] for d, cohs in DISEASE.items()}
