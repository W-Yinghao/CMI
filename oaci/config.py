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
    """Risk-feasible noninferiority constraint (THEORY §2): min UCB[L_Q^ov] s.t. R_src<=R_ERM+ε."""
    epsilon: float = 0.01            # source-risk slack over the ERM lower bound R_ERM
    risk_metric: str = "balanced_ce"  # 'ce' | 'balanced_ce' | 'balanced_err'
    lexicographic: bool = True       # stage 1: fit ERM for R_ERM; stage 2: constrained min
    # primal–dual multiplier on the leakage term (Lagrange knob; not a modelling choice)
    dual_lr: float = 0.5
    lambda_init: float = 0.0
    lambda_max: float = 20.0
    constraint_warmup: int = 5       # epochs before the dual variable starts moving


@dataclass
class OACIConfig:
    """Top-level OACI configuration."""
    support: SupportConfig = field(default_factory=SupportConfig)
    ucb: UCBConfig = field(default_factory=UCBConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
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
        return self


if __name__ == "__main__":
    cfg = OACIConfig().validate()
    print("OACIConfig OK:")
    print("  support:", cfg.support)
    print("  ucb    :", cfg.ucb)
    print("  risk   :", cfg.risk)
