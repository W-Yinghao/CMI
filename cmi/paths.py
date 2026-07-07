"""Central path / environment configuration.

Importing this module sets the MNE / MOABB environment variables so that all
MOABB datasets resolve to a local datalake cache *offline* (no download).
Paths are overridable by environment variable so the code is site-independent.
"""
import os
from pathlib import Path

# Project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Read-only datalake (MOABB/MNE cache). Override with EEG_DATALAKE_RAW.
DATALAKE_RAW = Path(os.environ.get("EEG_DATALAKE_RAW", str(PROJECT_ROOT / "data" / "datalake" / "raw")))

# Writable scratch/download area (checked before any download). Override with EEG_SCRATCH.
SCRATCH = Path(os.environ.get("EEG_SCRATCH", str(PROJECT_ROOT / "scratch")))


def configure_offline_moabb() -> None:
    """Point MNE/MOABB at the local datalake and disable downloads."""
    root = str(DATALAKE_RAW)
    os.environ.setdefault("MNE_DATA", root)
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", root)
    os.environ.setdefault("MNE_DATASETS_GIGADB_PATH", root)
    os.environ.setdefault("MNE_DATASETS_EEGBCI_PATH", root)
    os.environ.setdefault("MNE_DATASETS_SCHIRRMEISTER2017_PATH", root)
    # MOABB respects MNE_DATA for most datasets.
    os.environ.setdefault("MOABB_RESULTS", str(RESULTS_DIR / "moabb"))


# Motor-imagery dataset registry: MOABB class name -> metadata.
MI_DATASETS = {
    "BNCI2014_001": dict(n_subjects=9,  classes=4, sessions=2,  fs=250,  ch=22, note="BCI-IV-2a"),
    "BNCI2014_004": dict(n_subjects=9,  classes=2, sessions=5,  fs=250,  ch=3,  note="BCI-IV-2b (L/R)"),
    "Lee2019_MI":   dict(n_subjects=54, classes=2, sessions=2,  fs=1000, ch=62, note="OpenBMI L/R"),
    "Cho2017":      dict(n_subjects=52, classes=2, sessions=1,  fs=512,  ch=64, note="GigaScience L/R"),
    "Schirrmeister2017": dict(n_subjects=14, classes=4, sessions=1, fs=500, ch=128, note="High-Gamma"),
    "Stieger2021":  dict(n_subjects=62, classes=4, sessions=11, fs=1000, ch=64, note="multi-session"),
    "Weibo2014":    dict(n_subjects=10, classes=7, sessions=1,  fs=200,  ch=60, note="supplement"),
    "Zhou2016":     dict(n_subjects=4,  classes=3, sessions=3,  fs=250,  ch=14, note="supplement"),
}

if __name__ == "__main__":
    configure_offline_moabb()
    print("DATALAKE_RAW exists:", DATALAKE_RAW.exists())
    print("env MNE_DATA:", os.environ.get("MNE_DATA"))
