"""Dataset registry + offline guard. Loaders are OFFLINE-ONLY (read the read-only datalake; never
download). The unit definitions per paradigm are fixed here (MI: trial = support = eval unit,
group = recording; SEED: window aggregated to film-clip eval unit; clinical: window aggregated to
subject eval unit, domain = site/cohort, group = subject).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DATALAKE = "/projects/EEG-foundation-model/datalake/raw"


class OfflineDownloadError(RuntimeError):
    """Raised instead of downloading when offline data is missing."""


@dataclass
class DatasetEntry:
    id: str
    paradigm: str                 # 'MI' | 'affective' | 'clinical_rest'
    loader: str                   # 'moabb' | 'seed' | 'clinical_bids'
    domain_factor: str            # 'subject_id' (MI/SEED LOSO) | 'site_id' (clinical)
    group_factor: str = "recording_id"
    support_unit: str = "trial_id"
    eval_unit: str = "trial_id"
    moabb_id: str = ""
    classes: tuple = ()
    offline_root: str = ""
    cohorts: tuple = ()           # clinical: multiple same-paradigm cohorts (sites)
    notes: str = ""


REGISTRY: dict[str, DatasetEntry] = {
    "BNCI2014_001": DatasetEntry("BNCI2014_001", "MI", "moabb", "subject_id",
                                 moabb_id="BNCI2014_001", classes=("left_hand", "right_hand", "feet", "tongue"),
                                 offline_root=DATALAKE),
    "BNCI2014_004": DatasetEntry("BNCI2014_004", "MI", "moabb", "subject_id",
                                 moabb_id="BNCI2014_004", classes=("left_hand", "right_hand"),
                                 offline_root=DATALAKE),
    "Cho2017": DatasetEntry("Cho2017", "MI", "moabb", "subject_id",
                            moabb_id="Cho2017", classes=("left_hand", "right_hand"), offline_root=DATALAKE),
    "Lee2019_MI": DatasetEntry("Lee2019_MI", "MI", "moabb", "subject_id",
                               moabb_id="Lee2019_MI", classes=("left_hand", "right_hand"), offline_root=DATALAKE),
    "SEED": DatasetEntry("SEED", "affective", "seed", "subject_id",
                         support_unit="eval_unit_id", eval_unit="eval_unit_id",
                         classes=("negative", "neutral", "positive"), offline_root=DATALAKE,
                         notes="window aggregated to film-clip eval unit"),
    "PD_cross_site": DatasetEntry("PD_cross_site", "clinical_rest", "clinical_bids", "site_id",
                                  group_factor="subject_id", support_unit="subject_id", eval_unit="subject_id",
                                  classes=("HC", "PD"), offline_root=DATALAKE,
                                  notes="MAIN PD: restrict to same-paradigm (resting) cohorts; LOSO over 3 resting sites"),
    "SCZ_cross_site": DatasetEntry("SCZ_cross_site", "clinical_rest", "clinical_bids", "site_id",
                                   group_factor="subject_id", support_unit="subject_id", eval_unit="subject_id",
                                   classes=("HC", "SCZ"), offline_root=DATALAKE,
                                   notes="only 2 resting cohorts -> rest-only LOSO leaves 1 source site/fold: "
                                         "method-inactive (no-op), NOT confirmatory efficacy evidence"),
}


def get_entry(dataset_id: str) -> DatasetEntry:
    if dataset_id not in REGISTRY:
        raise KeyError(f"unknown dataset {dataset_id!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id]


def set_offline_env() -> None:
    """Point MNE/MOABB at the read-only datalake; do not enable any download."""
    os.environ.setdefault("MNE_DATA", DATALAKE)
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", DATALAKE)
    os.environ.setdefault("MNE_DATASETS_GIGADB_PATH", DATALAKE)
    os.environ.setdefault("MNE_DATASETS_LEE2019_MI_PATH", DATALAKE)
    os.environ.setdefault("MOABB_OFFLINE", "1")


def ensure_offline_available(path: str) -> str:
    """Raise (never download) if the offline data path is missing."""
    if not path or not os.path.exists(path):
        raise OfflineDownloadError(f"offline data not found at {path!r}; refusing to download")
    return path
