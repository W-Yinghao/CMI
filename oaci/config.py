"""Configuration dataclasses for OACI (Overlap-Aware Risk-Feasible Conditional Invariance).

One ``OACIConfig`` parameterises the three moving parts of the method:

  * ``support``   — what counts as estimable support (the support graph threshold ``m``);
  * ``ucb``       — how the *upper confidence bound* on the probe-class-extractable leakage
                    ``L_Q^ov`` is formed (cross-fit critic, domain/class-clustered bootstrap,
                    multi-capacity probe, confidence level ``1-α``). This UCB is a genuine
                    statistical bound on ``L_Q^ov`` (a LOWER bound on true ``I_ov`` — see
                    ``UCBConfig`` / THEORY §4), NOT a posterior-KL relabelled as one (the
                    AAAI/LPC error).
  * ``risk``      — the risk-feasibility constraint ``R_src(θ) <= R_ERM + ε`` and the
                    primal–dual multiplier that enforces it. ``λ`` is a Lagrange knob, not
                    part of the model's meaning.

Defaults are sized to RUN small (smoke / synthetic support patterns); scale for real EEG.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SupportConfig:
    """Domain–class support graph (THEORY §1)."""
    m_min: int = 20                  # min effective count for a (d,y) cell to be estimable
    use_effective_n: bool = True     # treat counts as effective sample sizes (within-domain
    #                                  trial correlation) rather than raw trial counts
    min_domains_per_class: int = 2   # |S_y| >= this for a class to contribute a leakage term
    # cells below m_min are NEVER smoothed/imputed; they are reported as non-identifiable.


@dataclass
class UCBConfig:
    """Upper confidence bound on the *probe-class-extractable* overlap leakage L_Q^ov.

    IMPORTANT (THEORY §4). A finite probe ``q`` gives
    ``E[-log q(D|Z,Y)] = H(D|Z,Y) + E_{Z,Y} KL(p(·|Z,Y) ‖ q)``, so the probe entropy-gap
    ``Ĥ(D|Y) - E[-log q]`` is a *lower* bound on the true ``I_ov`` — an imperfect probe
    UNDERESTIMATES leakage. We therefore do NOT bound ``I_ov`` from above. The operational
    functional we minimise and bound is

        ``L_Q^ov = sup_{q∈Q} ( Ĥ(D|Y) - E[-log q(D|Z,Y)] )``   over comparable cells,

    the leakage extractable by the fixed probe family ``Q`` ("extractable conditional domain
    information", per the project's binding naming). ``UCB_{1-α}[L_Q^ov]`` is an upper
    confidence bound on THAT functional, from a clustered bootstrap. The capacity sup over
    ``Q`` (``probe_capacities``) is part of the estimator, so it must be re-selected INSIDE
    every bootstrap resample (``capacity_selection_in_resample``) or the bound is optimistic.
    """
    functional: str = "extractable_LQ_ov"               # the estimand we bound (NOT I_ov)
    alpha: float = 0.1                                  # UCB level -> report the (1-α) bound
    cross_fit_folds: int = 5                            # held-out critic estimate (no upward bias)
    n_bootstrap: int = 200                              # clustered bootstrap resamples
    cluster_by: str = "recording"                       # dependence unit, resampled WITHIN domain: 'recording' (default) | 'domain' | 'class' | 'domain_class'
    probe_capacities: tuple[int, ...] = (32, 128, 512)  # the probe family Q (sup over these)
    capacity_selection_in_resample: bool = True         # re-select capacity per resample (else biased)
    weighting: str = "absolute"                         # 'absolute' (L_abs, fixed p_ref) | 'conditional' (L_cond, diag)
    keep_negative: bool = True                          # do NOT truncate negative excess (bias)
    n_perm_null: int = 100                              # within-(Y,d-group) permutation null


@dataclass
class RiskConfig:
    """Risk-feasible noninferiority constraint (THEORY §2): min UCB[L_Q^ov] s.t. R_src<=τ."""
    # τ = R̂_ERM + ε, where R̂_ERM is the REALISED empirical risk of the frozen Stage-1
    # checkpoint (NOT an "ERM lower bound"). ε is the noninferiority slack over R̂_ERM.
    epsilon: float = 0.01            # source-risk slack over the realised Stage-1 risk R̂_ERM
    risk_metric: str = "balanced_ce"  # 'ce' | 'balanced_ce' | 'balanced_err'
    lexicographic: bool = True       # stage 1: fit ERM for R̂_ERM; stage 2: constrained min
    # dual multiplier of the RISK CONSTRAINT R_src <= τ (NOT a weight on the leakage term):
    # λ ← Π_[0,λmax]( λ + η_λ (R̂_guard − τ) ). It is a Lagrange knob, not a modelling choice.
    dual_lr: float = 0.5             # η_λ
    lambda_init: float = 1.0         # start > 0 so risk is anchored from the first encoder step
    lambda_max: float = 20.0
    constraint_warmup: int = 5       # critic-only warmup (encoder frozen) before the min–max game


@dataclass
class SamplerConfig:
    """Paired-stream rare-cell sampler. Eligibility, S_y, p_ref and n_{d,y} are FIXED (the
    full-data support graph); a batch only guarantees eligible-cell COVERAGE and never
    redefines eligibility. Importance weights restore the FIXED empirical priors, not the
    sampler-induced near-uniform ones (see `oaci/data/sampler.py`)."""
    task_batch_size: int = 256          # task stream: stratified by class; includes ineligible cells
    adv_microbatch_size: int = 256      # adversary-stream microbatch (memory cap)
    adv_accumulation_steps: int = 1     # microbatches per logical adv batch (optimiser steps once)
    min_per_eligible_cell: int = 8      # k_min: rows drawn per eligible (d,y) cell per logical batch
    steps_per_epoch: int = 20
    replacement_mode: str = "auto"      # 'auto' (no repeat until exhausted) | 'always' | 'never'
    seed: int = 0

    def validate(self) -> "SamplerConfig":
        if self.replacement_mode not in {"auto", "always", "never"}:
            raise ValueError(f"replacement_mode must be auto|always|never, got {self.replacement_mode!r}")
        for f in ("task_batch_size", "adv_microbatch_size", "adv_accumulation_steps", "min_per_eligible_cell"):
            if getattr(self, f) < 1:
                raise ValueError(f"SamplerConfig.{f} must be >= 1")
        return self

    def assert_capacity(self, n_eligible_cells: int) -> None:
        """B_min = K_ov · k_min must fit in adv_microbatch_size · adv_accumulation_steps."""
        b_min = n_eligible_cells * self.min_per_eligible_cell
        cap = self.adv_microbatch_size * self.adv_accumulation_steps
        if cap < b_min:
            raise ValueError(
                f"adversary capacity {cap} (= adv_microbatch_size {self.adv_microbatch_size} x "
                f"adv_accumulation_steps {self.adv_accumulation_steps}) < logical batch B_min {b_min} "
                f"(= K_ov {n_eligible_cells} x k_min {self.min_per_eligible_cell}). "
                f"Raise the capacity or lower min_per_eligible_cell; cells must not be dropped."
            )


@dataclass
class EvalConfig:
    """Evaluation: fixed-estimand metrics + paired clustered bootstrap + noninferiority.

    ``delta_bacc`` (the target balanced-accuracy noninferiority margin) has NO default — it must
    be set explicitly per experiment so it is never silently reused / mistaken for ``ε`` (which
    is the source-risk slack only, and only when the audit metric matches the training metric).
    """
    n_ece_bins: int = 15                 # FIXED pre-registered equal-width bins, shared everywhere
    alpha: float = 0.05                  # one-sided CI level
    n_boot: int = 2000                   # paired clustered bootstrap replicates
    invalid_threshold: float = 0.2       # CI non-estimable if invalid_draw_rate exceeds this
    min_clusters: int = 2                # an eval domain needs >= this many recording groups
    delta_bacc: float | None = None      # target-bAcc NI margin — REQUIRED (no paper-threshold default)
    risk_metric: str = "balanced_ce"     # MUST equal the training constraint metric to reuse ε

    def require_delta_bacc(self) -> float:
        if self.delta_bacc is None:
            raise ValueError("EvalConfig.delta_bacc must be set explicitly (it is not ε; no default).")
        return float(self.delta_bacc)


@dataclass
class OACIConfig:
    """Top-level OACI configuration."""
    support: SupportConfig = field(default_factory=SupportConfig)
    ucb: UCBConfig = field(default_factory=UCBConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0

    def validate(self) -> "OACIConfig":
        if not 0.0 < self.ucb.alpha < 1.0:
            raise ValueError(f"ucb.alpha must be in (0,1), got {self.ucb.alpha}")
        if self.support.min_domains_per_class < 2:
            raise ValueError("a conditional leakage term needs >= 2 domains per class")
        if self.risk.epsilon < 0:
            raise ValueError("risk.epsilon (noninferiority slack) must be >= 0")
        if self.ucb.cluster_by not in {"domain", "class", "domain_class", "recording"}:
            raise ValueError(f"unknown ucb.cluster_by: {self.ucb.cluster_by}")
        if self.ucb.weighting not in {"absolute", "conditional"}:
            raise ValueError(f"ucb.weighting must be 'absolute' or 'conditional', got {self.ucb.weighting}")
        self.sampler.validate()
        return self


if __name__ == "__main__":
    cfg = OACIConfig().validate()
    print("OACIConfig OK:")
    print("  support:", cfg.support)
    print("  ucb    :", cfg.ucb)
    print("  risk   :", cfg.risk)
