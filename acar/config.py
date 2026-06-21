"""Frozen constants and config for ACAR. Mirrors the substrate the A0 line was bound to so results are comparable.

These values are pre-registered in notes/ACAR_FROZEN.md and must not drift. The dataclass collects the few knobs
the go/no-go exposes on the CLI (alpha, delta, batch size, action set); everything substrate-level is frozen here.
"""
from dataclasses import dataclass, field
import os

# ---- substrate (frozen) ----
# erm_0 dumps were archived during the LPC-CMI closeout; prefer the live results/ copy if it reappears.
_V4_CANDIDATES = ["results/feat_dump_v4", "archive/lpc-cmi-failed/results/feat_dump_v4"]


def feat_dump_dir() -> str:
    for p in _V4_CANDIDATES:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "audit_PD_ds002778_erm_0.npz")):
            return p
    raise FileNotFoundError(
        "feat_dump_v4 erm_0 dumps not found in " + " or ".join(_V4_CANDIDATES))


DISEASE = {"PD": ["ds002778", "ds003490", "ds004584"],
           "SCZ": ["ds003944", "ds003947", "ds004000", "ds004367"]}
N_CLS = 2
RHO = 0.1            # source-state shrinkage; matches the A0 deployed config (CLI default 0.1, NOT manifest 0.2)

# ---- candidate actions (frozen set; identity is the f_0 reference, always present) ----
ACTIONS = ["identity", "matched_coral", "spdim", "t3a"]
NON_IDENTITY = [a for a in ACTIONS if a != "identity"]

# ---- paired label-free features (order is the φ_a vector order) ----
PAIRED_FEATURES = ["d_entropy", "d_margin", "flip_rate", "js", "bures", "post_sep", "n_eff"]
# A0 source-free scores carried as background context coordinates only (NO asserted direction).
CONTEXT_FEATURES = ["g_unc", "s_support", "s_sep", "pr_cmi_proxy"]

# ---- batching / fallback (frozen) ----
B = 32              # natural batch size, recording-ordered
MIN_BATCH = 8       # label-blind identity fallback below this (the ONLY fallback rule)

# ---- go/no-go thresholds (frozen) ----
AUROC_GATE = 0.60   # G1
RETAIN_FRAC = 0.50  # G2: keep >= this fraction of always-adapt's beneficial alignment


@dataclass
class ACARConfig:
    alpha: float = 0.10        # conformal miscoverage (1-alpha one-sided coverage)
    delta: float = 0.0         # act only when U_a < -delta
    batch: int = B
    seed: int = 0
    actions: list = field(default_factory=lambda: list(ACTIONS))
    risk: str = "nll"          # "nll" (primary) or "01" (secondary)
    out: str = "results/acar_gonogo"

    def __post_init__(self):
        assert self.actions[0] == "identity", "identity must be the f_0 reference action"
        assert self.risk in ("nll", "01")
