"""Configuration dataclasses for OACI (Overlap-Aware Risk-Feasible Conditional Invariance).

One ``OACIConfig`` parameterises the three moving parts of the method:

  * ``support``   — what counts as estimable support (the support graph threshold ``m``);
  * ``ucb``       — how the *upper confidence bound* on overlap-aware leakage is formed
                    (cross-fit critic, domain/class-clustered bootstrap, multi-capacity
                    probe, confidence level ``1-α``). This UCB is a genuine statistical
                    bound, NOT a posterior-KL relabelled as one (the AAAI/LPC error).
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
    """Upper confidence bound on the overlap-aware conditional leakage I_ov(Z;D|Y).

    The point estimate is a cross-fitted neural conditional-entropy gap
    ``Î_ov = H(D|Y) - H(D|Z,Y)`` over the *comparable* cells only; the uncertainty comes
    from a clustered bootstrap over the natural unit of dependence, and capacity is swept
    so the reported number is a max-probe (worst-case extractable) quantity.
    """
    alpha: float = 0.1                                  # UCB level -> report the (1-α) bound
    cross_fit_folds: int = 5                            # held-out critic estimate (no upward bias)
    n_bootstrap: int = 200                              # clustered bootstrap resamples
    cluster_by: str = "domain_class"                    # 'domain' | 'class' | 'domain_class' | 'recording'
    probe_capacities: tuple[int, ...] = (32, 128, 512)  # multi-capacity critic hidden sizes
    keep_negative: bool = True                          # do NOT truncate negative excess (bias)
    n_perm_null: int = 100                              # within-(Y,d-group) permutation null


@dataclass
class RiskConfig:
    """Risk-feasible noninferiority constraint (THEORY §2): min UCB[I_ov] s.t. R_src<=R_ERM+ε."""
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
        return self


if __name__ == "__main__":
    cfg = OACIConfig().validate()
    print("OACIConfig OK:")
    print("  support:", cfg.support)
    print("  ucb    :", cfg.ucb)
    print("  risk   :", cfg.risk)
