"""V2 real-acquisition metadata registry + frozen geometry/prevalence classifiers (review V2_FROZEN).

Derives the two metadata axes from ORIGINAL acquisition descriptors (not post-preprocessing
appearance), then routes through the UNCHANGED frozen rule table `metadata_to_operator`. The
rule table itself is NOT modified -- only the deployable axis derivation is new (exactly what the
V2 mapping requires). Two tightenings vs the simulator substrate:
  (1) cross-session SAME acquisition (different day, same system/montage/reference/device/task &
      feedback regime) -> DIAG_COMPATIBLE  (session/day drift IS the diagonal gain/calibration drift).
  (2) sampling/filter/window differences are PREPROCESSING harmonisation (RESOLVED), never adaptation
      evidence; common-channel intersection + resample do NOT downgrade reference/device/montage
      mismatch from UNSUPPORTED to DIAG_COMPATIBLE.
Prevalence: MI left/right is a documented balanced cue schedule -> SAME (never inferred from cohort
identity, predictions, occupancy, or target labels).
"""
from __future__ import annotations

from dataclasses import dataclass

from h2cmi.data.metadata_substrate import MetadataDelta, metadata_to_operator


def MOABB_CLASS(name=None):
    """Lazy MOABB dataset class registry (import only when needed; offline)."""
    from moabb.datasets import BNCI2014_001, BNCI2014_004, Cho2017, Lee2019_MI
    reg = {"BNCI2014_001": BNCI2014_001, "BNCI2014_004": BNCI2014_004,
           "Cho2017": Cho2017, "Lee2019_MI": Lee2019_MI}
    return reg if name is None else reg[name]


class _Reg(dict):
    def __call__(self, name):
        return self[name]
    def __getitem__(self, k):
        from moabb.datasets import BNCI2014_001, BNCI2014_004, Cho2017, Lee2019_MI
        return {"BNCI2014_001": BNCI2014_001, "BNCI2014_004": BNCI2014_004,
                "Cho2017": Cho2017, "Lee2019_MI": Lee2019_MI}[k]


MOABB_CLASS = _Reg()


@dataclass(frozen=True)
class Acquisition:
    """Original acquisition descriptor for one dataset (deployable, label-free)."""
    dataset: str
    device_family: str          # amplifier family
    reference: str              # reference scheme
    montage: str                # cap/electrode system
    measurement: str            # 'monopolar' | 'bipolar'
    task: str                   # 'LR_MI'
    n_eeg: int


# Frozen registry (public dataset documentation; review V2_FROZEN §1).
ACQ = {
    "BNCI2014_001": Acquisition("BNCI2014_001", "gtec_gusbamp", "left_mastoid", "10-20-22ch",
                                "monopolar", "LR_MI", 22),
    "Cho2017":      Acquisition("Cho2017", "biosemi_activetwo", "CMS_DRL", "10-10-64ch",
                                "monopolar", "LR_MI", 64),
    "Lee2019_MI":   Acquisition("Lee2019_MI", "brainamp", "nasion", "10-10-62ch",
                                "monopolar", "LR_MI", 62),
    "BNCI2014_004": Acquisition("BNCI2014_004", "gtec", "unknown", "bipolar-3ch-C3CzC4",
                                "bipolar", "LR_MI", 3),
}

# Feedback regime per (dataset, session-index). BNCI2014_004 sessions 0,1 = screening (no feedback),
# 2,3,4 = feedback; crossing them changes the conditional -> UNSUPPORTED (so 2->3 is NOT run).
def feedback_regime(dataset: str, session: int) -> str:
    if dataset == "BNCI2014_004":
        return "screening" if session in (0, 1) else "feedback"
    return "standard"


def v2_geometry(src_ds: str, src_sess: int, tgt_ds: str, tgt_sess: int) -> str:
    """DIAG_COMPATIBLE / UNSUPPORTED / UNKNOWN from ORIGINAL acquisition (never NONE here: V2 only
    runs genuine cross-session or cross-dataset shifts)."""
    if src_ds not in ACQ or tgt_ds not in ACQ:
        return "UNKNOWN"
    a, b = ACQ[src_ds], ACQ[tgt_ds]
    if src_ds != tgt_ds:                                  # cross-dataset
        # any acquisition-geometry mismatch the diagonal family cannot transport -> UNSUPPORTED
        if (a.reference != b.reference or a.montage != b.montage or a.device_family != b.device_family
                or a.measurement != b.measurement):
            return "UNSUPPORTED"
        return "DIAG_COMPATIBLE"                          # (does not occur among the V2 A datasets)
    # same dataset: feedback/task regime change -> UNSUPPORTED; else cross-session drift -> DIAG
    if feedback_regime(src_ds, src_sess) != feedback_regime(tgt_ds, tgt_sess):
        return "UNSUPPORTED"
    return "DIAG_COMPATIBLE"


def v2_prevalence(src_ds: str, tgt_ds: str) -> str:
    """All V2 datasets document a balanced L/R cue schedule -> SAME (cue-schedule, not cohort)."""
    return "SAME"


def v2_delta(src_ds: str, src_sess: int, tgt_ds: str, tgt_sess: int) -> MetadataDelta:
    """Build a MetadataDelta with the two derived axes; raw fields set for audit (original metadata)."""
    same_ds = src_ds == tgt_ds
    a = ACQ.get(src_ds); b = ACQ.get(tgt_ds)
    d = MetadataDelta(
        device_same=(a.device_family == b.device_family) if (a and b) else None,
        reference_same=(a.reference == b.reference) if (a and b) else None,
        montage_same=(a.montage == b.montage) if (a and b) else None,
        channel_layout_same=(a.montage == b.montage) if (a and b) else None,
        sampling_filter_same=True,                        # RESOLVED by frozen preprocessing
        cohort_same=same_ds, sampling_protocol_same=True)
    d.geometry_compatibility = v2_geometry(src_ds, src_sess, tgt_ds, tgt_sess)
    d.prevalence_risk = v2_prevalence(src_ds, tgt_ds)
    return d


def v2_operator(src_ds: str, src_sess: int, tgt_ds: str, tgt_sess: int) -> tuple[str, MetadataDelta]:
    """Frozen rule-table decision: returns (operator_name, delta)."""
    d = v2_delta(src_ds, src_sess, tgt_ds, tgt_sess)
    return metadata_to_operator(d), d
