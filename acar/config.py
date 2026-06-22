"""Frozen constants and config for ACAR. Substrate mirrors the A0 line so results are comparable.

v2 (notes/ACAR_FROZEN_v2.md): calibration unit = subject/recording cluster (NOT cohort); disease-stratified
subject-clustered split conformal. Substrate-level values are frozen here; the CLI exposes only alpha/delta/seed.
"""
from dataclasses import dataclass, field
import os

# ---- substrate (frozen) ----
# erm_0 dumps were archived during the LPC-CMI closeout. Resolution order: $ACAR_FEAT_DUMP, then relative candidates
# (the npz are gitignored, so a worktree off exp/lpc-cmi needs the env var or a symlink to the main checkout's dumps).
_V4_CANDIDATES = ["results/feat_dump_v4", "archive/lpc-cmi-failed/results/feat_dump_v4",
                  "/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4"]


def feat_dump_dir() -> str:
    cands = ([os.environ["ACAR_FEAT_DUMP"]] if os.environ.get("ACAR_FEAT_DUMP") else []) + _V4_CANDIDATES
    for p in cands:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "audit_PD_ds002778_erm_0.npz")):
            return p
    raise FileNotFoundError("feat_dump_v4 erm_0 dumps not found; set $ACAR_FEAT_DUMP. Tried: " + " | ".join(cands))


DISEASE = {"PD": ["ds002778", "ds003490", "ds004584"],
           "SCZ": ["ds003944", "ds003947", "ds004000", "ds004367"]}
N_CLS = 2
RHO = 0.1            # source-state shrinkage; matches the A0 deployed config (CLI default 0.1)

# ---- candidate actions (identity is the f_0 reference, always present) ----
ACTIONS = ["identity", "matched_coral", "spdim", "t3a"]
NON_IDENTITY = [a for a in ACTIONS if a != "identity"]

# ---- paired label-free features (φ_a vector order) ----
PAIRED_FEATURES = ["d_entropy", "d_margin", "flip_rate", "js", "bures", "post_sep", "n_eff"]
CONTEXT_FEATURES = ["g_unc", "s_support", "s_sep", "pr_cmi_proxy"]   # A0 scores as context only (no asserted dir)

# ---- batching / fallback (frozen) ----
B = 32              # natural batch size, recording-ordered
MIN_BATCH = 8       # label-blind: batches below this are RETAINED but forced to identity (v2; v1 wrongly deleted)

# ---- v2 subject-clustered conformal (frozen) ----
K_FOLDS = 5         # subject-disjoint CV folds (every subject is EVAL out-of-fold once)
FIT_FRAC = 0.70     # of non-EVAL subjects: FIT vs CAL split (subject-disjoint)

# ---- go/no-go thresholds (frozen) ----
AUROC_GATE = 0.60   # G1
RETAIN_FRAC = 0.50  # G2: oracle benefit-retention floor


@dataclass
class ACARConfig:
    alpha: float = 0.10        # conformal miscoverage (1-alpha one-sided coverage)
    delta: float = 0.0         # act only when U_a < -delta
    batch: int = B
    seed: int = 0
    actions: list = field(default_factory=lambda: list(ACTIONS))
    risk: str = "nll"          # "nll" (primary) or "01" (secondary)
    k_folds: int = K_FOLDS
    out: str = "results/acar_gonogo"

    def __post_init__(self):
        assert self.actions[0] == "identity", "identity must be the f_0 reference action"
        assert self.risk in ("nll", "01")

    def config_hash(self):
        import hashlib, json
        d = dict(alpha=self.alpha, delta=self.delta, batch=self.batch, seed=self.seed,
                 actions=self.actions, risk=self.risk, k_folds=self.k_folds,
                 paired=PAIRED_FEATURES, context=CONTEXT_FEATURES, rho=RHO, min_batch=MIN_BATCH)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
