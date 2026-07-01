"""ACAR V5 protocol constants — the SINGLE SOURCE OF TRUTH mirroring the frozen docs (tag acar-v5-protocol @ 4278435):
notes/ACAR_FROZEN_v5.md · ACAR_V5_CANDIDATE_SPACE.md · ACAR_V5_ENDPOINTS.md · ACAR_V5_SPLITS.md. Pure data + a self-check; NO I/O.

Any change here that diverges from the tagged docs is a protocol violation, not a refactor. The guard tests import THESE values
so a silent drift between code and the frozen protocol turns a test red.
"""
from __future__ import annotations

# ---- actions (z-space, GPU-free) + tie-break order (matched_coral ≺ spdim ≺ t3a); identity = abstain -----------------
IDENTITY = "identity"
ACTIONS = ("matched_coral", "spdim", "t3a")                       # non-identity actions
ACTION_ORDER = {a: i for i, a in enumerate(ACTIONS)}             # deterministic argmax/argmin/agreement tie-break

# ---- label-free paired features (routing may read ONLY these; never labels) ------------------------------------------
FEATURES = ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")

# ---- FIT-only quantile grid (the ONLY allowed operating quantiles) ---------------------------------------------------
QUANTILE_GRID = ("q50", "q60", "q70", "q80", "q85", "q90")
QUANTILE_VALUE = {"q50": 0.50, "q60": 0.60, "q70": 0.70, "q80": 0.80, "q85": 0.85, "q90": 0.90}

# ---- primary gates (ENDPOINTS §1) ------------------------------------------------------------------------------------
COVERAGE_MIN = 0.15                 # G1: LCB[coverage] ≥ 0.15
UTILITY_EPS = 0.02                  # G2: red − v2_replay ≥ 0.02 (effect size; not CI-only)
L_HARM_ALL_MAX = 0.10              # G3: UCB[L_harm_all] ≤ 0.10
HARM_AMONG_ADAPTED_MAX = 0.30      # G4: UCB[harm_among_adapted] ≤ 0.30  (the v4-missing gate)
BENEFIT_RETENTION_FRAC = 0.25      # G5: red ≥ 0.25 × red_upper  OR ≥ best-fixed-abstain
ALPHA = 0.05                       # Holm family-wise level for H1–H3 (empirical-Bernstein)

# ---- split / seeds (SPLITS §5, Step 2c/2d/2e) ------------------------------------------------------------------------
OUTER_K = 5
FIT_FRAC, CAL_FRAC = 0.70, 0.30
TRAIN_FRAC, VAL_FRAC = 0.80, 0.20
SPLIT_SALT = "ACAR_V5_SPLIT_V1"
SELECTION_SEED = 20260711                      # canonical Stage-2 DEV-selection encoder seed
S1_SEEDS = (20260711, 20260712, 20260713)      # S1 robustness (12/13 NEVER used for selection)
BOOTSTRAP_SEED = 20260714                       # reported (non-gating) LCB[red − v2_replay]

# ---- cohorts / external (SPLITS §3–§4) -------------------------------------------------------------------------------
DEV_COHORTS = {"PD": ("ds002778", "ds003490", "ds004584"),
               "SCZ": ("ds003944", "ds003947", "ds004000", "ds004367")}
EXTERNAL_PRIMARY = {"SCZ": "zenodo14808296", "PD": "ds007526"}   # single-site per disease
EXTERNAL_PROVISIONAL_NOT_ADMITTED = ("zenodo14178398",)          # ASZED — dated amendment only
EXTERNAL_EXCLUDED = ("ds007020",)

# ---- Stage-0 registry hash set (SPLITS §2): no hash ⇒ inadmissible for selection ------------------------------------
REGISTRY_HASH_FIELDS = (
    "encoder_state_dict_sha256", "encoder_checkpoint_file_sha256",
    "source_state_artifact_sha256", "source_state_file_sha256",
    "preprocessing_config_sha256", "feat_dump_sha256",
)
REGISTRY_META_FIELDS = (
    "channel_montage", "sampling_rate", "windowing_config",
    "cohort_inclusion_list", "random_seed", "git_commit", "env_lock_sha256",
)

FAMILIES = ("P1", "P2", "P3", "P4", "P5")


def build_candidate_manifest():
    """The EXACT 22-row Stage-2 candidate manifest (CANDIDATE_SPACE §1.6). Every candidate has disease_scope='both'
    (selected JOINTLY; per-disease FIT quantiles only). No add/remove/reorder without a dated pre-run amendment."""
    m = []
    for b in ("q60", "q80"):                              # P1: benefit d_margin + harm veto
        for v in ("q80", "q90"):
            m.append({"id": f"V5-P1-{len(m) % 4 + 1:03d}", "family": "P1", "disease_scope": "both",
                      "params": {"benefit_q": b, "veto_q": v}})
    p2 = []
    for c in ("q60", "q80"):                              # P2: low-violence confidence gate
        for v in ("q80", "q90"):
            p2.append({"id": f"V5-P2-{len(p2) + 1:03d}", "family": "P2", "disease_scope": "both",
                       "params": {"conf_q": c, "veto_q": v}})
    p3 = []
    for a in ACTIONS:                                     # P3: best-fixed action + abstention (also G5 comparator pool)
        for v in ("q80", "q90"):
            p3.append({"id": f"V5-P3-{len(p3) + 1:03d}", "family": "P3", "disease_scope": "both",
                       "params": {"action": a, "veto_q": v}, "comparator_role": "candidate+g5_best_fixed"})
    p4 = [{"id": f"V5-P4-{i + 1:03d}", "family": "P4", "disease_scope": "both",
           "params": {"k": k, "veto_q": "q90"}} for i, k in enumerate((2, 3))]   # benefit-score-variant agreement
    p5 = [{"id": f"V5-P5-{i + 1:03d}", "family": "P5", "disease_scope": "both",
           "params": {"lambda_q": q, "veto_q": "q90"}} for i, q in enumerate(QUANTILE_GRID)]   # direct-selective sweep
    out = m + p2 + p3 + p4 + p5
    return tuple(out)


CANDIDATE_MANIFEST = build_candidate_manifest()
CANDIDATE_IDS = tuple(c["id"] for c in CANDIDATE_MANIFEST)


def _self_check():
    """Fail-closed structural invariants of the tagged protocol (also asserted by the guard test)."""
    counts = {f: sum(1 for c in CANDIDATE_MANIFEST if c["family"] == f) for f in FAMILIES}
    assert counts == {"P1": 4, "P2": 4, "P3": 6, "P4": 2, "P5": 6}, counts
    assert len(CANDIDATE_MANIFEST) == 22, len(CANDIDATE_MANIFEST)
    assert len(set(CANDIDATE_IDS)) == 22, "candidate IDs must be unique"
    # every operating quantile is in the allowed grid (no q75, etc.)
    for c in CANDIDATE_MANIFEST:
        for k, val in c["params"].items():
            if k.endswith("_q") or k == "lambda_q":
                assert val in QUANTILE_GRID, f"{c['id']}: quantile {val!r} outside the pinned grid"
    assert ACTION_ORDER == {"matched_coral": 0, "spdim": 1, "t3a": 2}
    return True


_self_check()
