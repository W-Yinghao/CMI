"""
csc.run_envelope — CSC-P1.5 DEVELOPMENT difficulty-envelope sweep (HARNESS ONLY).

WHAT THIS IS
------------
Power/control of the concept-shift certificate is NOT a single number; it is a SURFACE over the
difficulty of the problem. This module maps that surface: for each cell of a difficulty grid it
runs the FROZEN protocol over many INDEPENDENT source-target clusters and reports an
OPERATING-REGION block (not an accuracy table):

    false-certification control  x  visible-concept power  x  abstention behaviour
                                 x  gate-failure decomposition

CRITICAL CONTRACT (reviewer P1.4.5 -> P1.5):
  * The denominator of EVERY rate/Clopper-Pearson bound is the number of INDEPENDENT
    source-target CLUSTERS (one fresh source seed per cluster), NOT the number of correlated
    targets generated under a single source. Each cluster contributes exactly ONE Bernoulli to
    each cluster-level endpoint (any-forbidden, fired, any-false-concept).
  * These are DEVELOPMENT seeds. The resulting map MAY NOT be used to select thresholds, define
    the operating region, or seed a confirmatory claim. Freezing requires a separate, previously
    UNSEEN cluster set. NO FREEZE / NO CONFIRMATORY / NO P2 is gated on this file.

This file does NOT run on import and does NOT run a sweep unless `--run` is passed explicitly.
Without `--run` the CLI only PRINTS the design (axes, baseline, grid, per-cell metric list).

DIFFICULTY AXES (reviewer order) and their simulator knobs
----------------------------------------------------------
  1. concept_effect_size      -> make_target(concept_target_scale=)   deployment boundary movement
  2. n_subjects               -> make_source(subjects_per_domain=)    (x n_domains source subjects)
  3. epochs_per_subject       -> SimConfig(epochs_min=, epochs_max=)  cluster-size profile
  4. within_subject_corr      -> SimConfig(subject_tau=)              subject random-effect scale
  5. class_imbalance          -> SimConfig(prior_alpha=)             low alpha => skewed priors
  6. concept_eigengap_sep     -> make_source(concept_domains=)  PROXY: more concept domains with
                                 varying magnitude => richer class-residual spectrum / larger
                                 leading eigengap. NB a TRUE multi-concept-axis eigengap needs a
                                 geom extension (single w_concept today) -> flagged `proxy=True`.
  7. covariate_leakage        -> make_target(cov_target_scale=)       nuisance movement w/ the shift
  8. target_subjects          -> make_target(subjects=)               target cluster count
  9. mechanism_family         -> geom-seed family offset (robustness across latent geometries);
                                 per-kind scoring already spans covariate/boundary/label mechanisms.

Run (only when the reviewer has approved a DEVELOPMENT sweep):
    python -m csc.run_envelope --run --clusters 12 --out csc/results/envelope_dev.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
from dataclasses import dataclass, field

import numpy as np

from csc.protocol import (
    ProtocolConfig, execute_protocol, _cp_bound, _concept_failure_reason,
)
from csc.certificate import FORBIDDEN, UNIDENTIFIABLE, COVARIATE_COMPATIBLE, CONCEPT_SUSPECT
from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH

KINDS = list(_TRUTH)                                   # full taxonomy (all mechanisms)
MUST_ABSTAIN = ["clean", "pure_conditional", "label_shift", "label_covariate_mixed"]
# stable (generator-knows-boundary-fixed) targets used for the false-concept control endpoint
SYNTHETIC_NULL_KINDS = ["covariate", "clean"]
_FAMILY_STRIDE = 100_000                               # geom-family seed offset (axis 9)


# --------------------------------------------------------------------------------------
# one point of the difficulty grid -- every axis maps to an explicit simulator knob above.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvelopePoint:
    # axis 1: deployment concept movement (boundary_coupled target). source atlas strength tracks it.
    concept_effect_size: float = 14.0
    source_concept_scale: float = 4.0
    # axis 2: source biological subjects = subjects_per_domain * n_domains
    subjects_per_domain: int = 22
    n_domains: int = 8
    # axis 3: epochs/subject profile (unequal in [min,max])
    epochs_min: int = 8
    epochs_max: int = 22
    # axis 4: within-subject correlation (subject random-effect scale)
    within_subject_corr: float = 0.2
    # axis 5: class imbalance (Dirichlet conc.; LOW => skewed per-domain priors)
    prior_alpha: float = 4.0
    # axis 6: concept eigengap separation -- PROXY via #concept domains (see module docstring)
    concept_domains: int = 3
    # axis 7: covariate leakage accompanying the shift (target nuisance movement)
    covariate_leakage: float = 10.0
    source_cov_scale: float = 2.0
    # axis 8: target subject count
    target_subjects: int = 30
    # axis 9: mechanism / latent-geometry family (seed offset; robustness across geometries)
    mechanism_family: int = 0
    # held-fixed generative scalars (exposed so a cell can pin them; not swept by default)
    class_sep: float = 1.2

    def with_axis(self, axis: str, value) -> "EnvelopePoint":
        if axis not in {f.name for f in dataclasses.fields(self)}:
            raise KeyError(f"unknown difficulty axis {axis!r}")
        return dataclasses.replace(self, **{axis: value})


def _materialize(p: EnvelopePoint, src_seed: int):
    """Translate an EnvelopePoint into (SimConfig, source-kwargs, target-kwargs). The geom is
    derived inside make_source from the SimConfig seed; the mechanism_family offsets that seed so
    distinct families realise distinct latent geometries while clusters within a family stay
    independent (src_seed varies per cluster)."""
    seed = p.mechanism_family * _FAMILY_STRIDE + src_seed
    cfg = SimConfig(seed=seed, sep=p.class_sep, subject_tau=p.within_subject_corr,
                    epochs_min=p.epochs_min, epochs_max=p.epochs_max, prior_alpha=p.prior_alpha)
    src_kw = dict(n_domains=p.n_domains, concept_domains=p.concept_domains,
                  concept_scale=p.source_concept_scale, cov_scale=p.source_cov_scale,
                  subjects_per_domain=p.subjects_per_domain, seed=seed)
    tgt_kw = dict(concept_target_scale=p.concept_effect_size, cov_target_scale=p.covariate_leakage,
                  subjects=p.target_subjects)
    return cfg, src_kw, tgt_kw


# --------------------------------------------------------------------------------------
# one difficulty cell: K INDEPENDENT source clusters x the full taxonomy.
# --------------------------------------------------------------------------------------
def run_cell(point: EnvelopePoint, cfg: ProtocolConfig, n_clusters: int,
             base_seed: int = 0, tgt_seed_base: int = 7_000) -> dict:
    """Each cluster = ONE fresh source seed + one target per kind drawn from that source's geom.
    Every cluster-level endpoint counts ONE Bernoulli per cluster (independent denominator)."""
    per_cluster = []          # one record/cluster: source props + per-kind certificate states
    for k in range(n_clusters):
        src_seed = base_seed + k
        scfg, src_kw, tgt_kw = _materialize(point, src_seed)
        src = make_source(scfg, **src_kw)
        states, vis_fail_reason, src_props = {}, None, None
        for kind in KINDS:
            tb = make_target(kind, scfg, geom=src.geom, seed=tgt_seed_base + src_seed, **tgt_kw)
            out = execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg,
                                   src_group_ids=src.group_ids, tgt_group_ids=tb.group_ids,
                                   tgt_condition_ids=np.zeros(len(tb.Z), int), seed=src_seed)
            states[kind] = out["certificate"].state
            if src_props is None:                            # source analysis is per-source (same
                sa = out["analysis"]                         # for all kinds in this cluster)
                src_props = dict(source_status=sa.source_status,
                                 concept_evidenced=bool(sa.concept_evidenced),
                                 attribution_unreliable=bool(sa.attribution_unreliable))
            if kind == "boundary_coupled":
                vis_fail_reason = _concept_failure_reason(out, cfg.alpha)
        per_cluster.append(dict(states=states, vis_fail_reason=vis_fail_reason, **src_props))

    n = n_clusters

    def cl_rate(pred):                                       # per-cluster Bernoulli rate
        return sum(pred(r) for r in per_cluster) / n if n else None

    any_forbidden = [any(r["states"][kd] in FORBIDDEN[_TRUTH[kd]] for kd in KINDS) for r in per_cluster]
    any_ma = [any(r["states"][kd] in FORBIDDEN[_TRUTH[kd]] for kd in MUST_ABSTAIN) for r in per_cluster]
    any_false_concept = [any(r["states"][kd] == CONCEPT_SUSPECT for kd in SYNTHETIC_NULL_KINDS)
                         for r in per_cluster]
    n_forbidden, n_ma, n_fc = sum(any_forbidden), sum(any_ma), sum(any_false_concept)
    n_fired = sum(r["states"]["boundary_coupled"] == CONCEPT_SUSPECT for r in per_cluster)

    # gate-failure decomposition over visible (boundary_coupled) clusters that did NOT fire
    decomp = {}
    for r in per_cluster:
        decomp[r["vis_fail_reason"]] = decomp.get(r["vis_fail_reason"], 0) + 1

    # abstention rate over ALL (cluster,kind) cells (descriptive)
    n_cells = n * len(KINDS)
    abstain_cells = sum(st == UNIDENTIFIABLE for r in per_cluster for st in r["states"].values())

    return dict(
        point=dataclasses.asdict(point),
        eigengap_axis_is_proxy=True,                         # axis 6 uses concept_domains proxy
        n_independent_clusters=n,
        # ---- false-certification control (cluster-denominated, exact one-sided CP UPPER) ----
        any_forbidden_full_suite=n_forbidden,
        any_forbidden_full_suite_cp_upper=_cp_bound(n_forbidden, n, side="upper"),
        any_false_positive_must_abstain=n_ma,
        any_false_positive_must_abstain_cp_upper=_cp_bound(n_ma, n, side="upper"),
        false_concept_on_synthetic_null=n_fc,
        false_concept_on_synthetic_null_cp_upper=_cp_bound(n_fc, n, side="upper"),
        # ---- visible-concept power (exact one-sided CP LOWER) ----
        visible_concept_power=(n_fired / n if n else None),
        visible_concept_power_cp_lower=_cp_bound(n_fired, n, side="lower"),
        # ---- coverage / abstention ----
        covariate_compatible_coverage=cl_rate(lambda r: r["states"]["covariate"] == COVARIATE_COMPATIBLE),
        abstention_rate_all_cells=(abstain_cells / n_cells if n_cells else None),
        # ---- source/atlas availability (per cluster; source analysis is shared across kinds) ----
        source_valid_rate=cl_rate(lambda r: r["source_status"] == "VALID"),
        source_invalid_rate=cl_rate(lambda r: r["source_status"] != "VALID"),
        support_invalid_rate=cl_rate(lambda r: r["source_status"] == "INVALID_SUPPORT"),
        attribution_unassessable_rate=cl_rate(
            lambda r: r["source_status"] == "UNASSESSED_CONCEPT_ATTRIBUTION"),
        attribution_unstable_rate=cl_rate(
            lambda r: r["source_status"] == "UNSTABLE_CONCEPT_ATTRIBUTION"),
        atlas_availability=cl_rate(lambda r: r["concept_evidenced"]),
        # ---- gate-failure decomposition over visible non-firing clusters ----
        gate_failure_decomposition=decomp,
        robust_consensus_abstain=decomp.get("not_dominant_or_robust_consensus_abstain", 0),
        residual_T_not_sig=decomp.get("residual_T_not_sig", 0),
        geometric_maxstat_not_sig=decomp.get("geometric_maxstat_not_sig", 0),
    )


# --------------------------------------------------------------------------------------
# default grid: a "star" (one-axis-at-a-time from the baseline), NOT a full Cartesian product
# (9 axes x several levels would be intractable and is not needed to MAP the surface). Each cell
# is labelled by the axis it varies; the baseline appears once.
# --------------------------------------------------------------------------------------
_DEFAULT_LEVELS = {
    "concept_effect_size": [6.0, 10.0, 14.0, 20.0],
    "subjects_per_domain": [8, 14, 22, 30],
    "epochs_max": [12, 22, 40],                 # epochs_min fixed; widens the cluster-size profile
    "within_subject_corr": [0.0, 0.2, 0.5, 1.0],
    "prior_alpha": [0.5, 1.0, 4.0],             # lower => more class imbalance
    "concept_domains": [1, 3, 5],               # axis-6 eigengap proxy
    "covariate_leakage": [2.0, 6.0, 10.0, 16.0],
    "target_subjects": [10, 20, 30, 50],
    "mechanism_family": [0, 1, 2],
}


def default_grid(baseline: EnvelopePoint = None):
    """Star design from `baseline`: the baseline once, then every (axis, level) that differs from
    it. Returns [(cell_label, EnvelopePoint), ...]."""
    base = baseline or EnvelopePoint()
    cells = [("baseline", base)]
    for axis, levels in _DEFAULT_LEVELS.items():
        for lv in levels:
            pt = base.with_axis(axis, lv)
            if pt != base:
                cells.append((f"{axis}={lv}", pt))
    return cells


def run_envelope(cells, cfg: ProtocolConfig, n_clusters: int, out: str = None,
                 base_seed: int = 0, quiet: bool = True) -> dict:
    """Execute the difficulty grid. Heavy: each cell = n_clusters sources x len(KINDS) targets x
    the full frozen protocol. ONLY call with reviewer approval for a DEVELOPMENT sweep."""
    if quiet:
        warnings.filterwarnings("ignore")
    grid = []
    for label, point in cells:
        block = run_cell(point, cfg, n_clusters, base_seed=base_seed)
        block["cell"] = label
        grid.append(block)
        print(f"[envelope] {label:28s} clusters={n_clusters} "
              f"forbidden={block['any_forbidden_full_suite']}/{n_clusters} "
              f"(CP-UB {block['any_forbidden_full_suite_cp_upper']:.3f}) "
              f"power={block['visible_concept_power']:.2f} "
              f"(CP-LB {block['visible_concept_power_cp_lower']:.3f}) "
              f"atlas={block['atlas_availability']:.2f}")
    payload = dict(
        kind="CSC-P1.5 DEVELOPMENT difficulty-envelope (operating-region map)",
        status="DEVELOPMENT — NOT a freeze sweep; results MAY NOT select thresholds / define the "
               "operating region / seed a confirmatory claim. Denominator = independent source-"
               "target clusters. Freeze needs a separate UNSEEN cluster set.",
        manifest_hash=cfg.hash(), protocol_manifest=cfg.manifest(),
        n_clusters_per_cell=n_clusters, base_seed=base_seed,
        difficulty_axes=list(_DEFAULT_LEVELS), eigengap_axis_is_proxy=True,
        n_cells=len(grid), grid=grid)
    if out:
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[envelope] wrote {out}")
    return payload


def _describe(cfg, cells, n_clusters):
    print("=== CSC-P1.5 DEVELOPMENT difficulty-envelope — DESIGN (dry run, nothing executed) ===")
    print(f"manifest={cfg.hash()[:12]}  cells={len(cells)}  clusters/cell={n_clusters}  "
          f"kinds/cluster={len(KINDS)}")
    print(f"protocol calls if run = {len(cells) * n_clusters * len(KINDS)}")
    print("\ndifficulty axes (-> simulator knob), star design from baseline:")
    for ax, lv in _DEFAULT_LEVELS.items():
        print(f"  {ax:22s} levels={lv}")
    print("\nbaseline EnvelopePoint:")
    for f in dataclasses.fields(EnvelopePoint):
        print(f"  {f.name:22s} = {getattr(EnvelopePoint(), f.name)}")
    print("\nper-cell metric block (every rate denominated by INDEPENDENT clusters):")
    for m in ("n_independent_clusters", "any_forbidden_full_suite(+CP UB)",
              "false_concept_on_synthetic_null(+CP UB)", "visible_concept_power(+CP LB)",
              "covariate_compatible_coverage", "abstention_rate_all_cells",
              "source_invalid_rate", "support_invalid_rate",
              "attribution_unassessable_rate", "attribution_unstable_rate",
              "atlas_availability", "gate_failure_decomposition",
              "robust_consensus_abstain", "residual_T_not_sig", "geometric_maxstat_not_sig"):
        print(f"  - {m}")
    print("\nNOTE: DEVELOPMENT only. Pass --run to execute (reviewer approval required); the map "
          "CANNOT define the operating region or seed a confirmatory claim.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CSC-P1.5 difficulty-envelope (DEVELOPMENT).")
    ap.add_argument("--run", action="store_true",
                    help="actually execute the sweep (default: only print the design)")
    ap.add_argument("--clusters", type=int, default=12, help="independent source clusters per cell")
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=40)
    ap.add_argument("--n_dir_boot", type=int, default=120)
    ap.add_argument("--target_n_boot", type=int, default=120)
    ap.add_argument("--tau_n_pseudotargets", type=int, default=240)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = ProtocolConfig(n_boot=args.n_boot, n_dir_boot=args.n_dir_boot,
                         target_n_boot=args.target_n_boot,
                         tau_n_pseudotargets=args.tau_n_pseudotargets)
    cfg.validate()
    cells = default_grid()
    if not args.run:
        _describe(cfg, cells, args.clusters)
    else:
        print("[envelope] DEVELOPMENT sweep — results may NOT select params / define operating "
              "region / seed confirmatory. Denominator = independent clusters.")
        run_envelope(cells, cfg, args.clusters, out=args.out, base_seed=args.base_seed)
