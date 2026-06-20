"""Configuration dataclasses for the H2-CMI pipeline.

One ``H2Config`` parameterises encoder, density head, hierarchical-CMI budgets,
reference-marginal alignment, disentanglement, SSL aux, TTA and the safety gate.
Defaults are tuned to RUN (small) on the synthetic EEG simulator; scale up for real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """EEG encoder (review 5.2): temporal + SPD + electrode-graph branches -> (z_c, z_n)."""
    n_chans: int = 16
    n_times: int = 256
    fs: float = 128.0
    # frequency bands (Hz) for the filterbank temporal branch and the SPD branch
    bands: tuple[tuple[float, float], ...] = ((4, 8), (8, 13), (13, 30), (30, 45))
    temporal_filters: int = 8           # temporal conv filters per band
    spd_rank: int = 8                   # BiMap output dim per band (SPD branch)
    graph_hidden: int = 16              # electrode set/graph encoder hidden
    fuse_hidden: int = 128
    z_c_dim: int = 32                   # class/task latent dimension
    z_n_dim: int = 16                   # nuisance/domain latent dimension
    dropout: float = 0.25
    use_temporal: bool = True
    use_spd: bool = True
    use_graph: bool = True
    cov_shrinkage: float = 0.05         # Ledoit-Wolf-style ridge on per-band covariance
    canonicalizer: bool = True          # constrained near-identity canonicalizer on z_c


@dataclass
class DensityConfig:
    """Class-conditional latent density head (review 5.3): Student-t mixture p(z_c|y)."""
    n_components: int = 1               # K per class
    cov_rank: int = 4                   # low-rank + diagonal covariance
    df: float = 8.0                     # Student-t degrees of freedom (robust to artifacts)
    eig_floor: float = 1e-2             # eigenvalue floor / variance floor
    ema: float = 0.0                    # 0 = full-grad; >0 = EMA prototype/cov update
    beta_density: float = 1.0           # weight on -log p(z_c|y) in the hybrid loss
    gamma_consistency: float = 0.5      # weight on JS(discriminative || generative)
    init_scale: float = 1.0


@dataclass
class CMIConfig:
    """Hierarchical neural conditional-entropy CMI (review 5.4 / P0-2)."""
    critic_hidden: int = 128
    critic_inner: int = 1               # Step-A critic updates per Step-B encoder update
    critic_lr: float = 2e-3
    # The penalty form +lambda*(H_ref - CE), minimised, already MAXIMISES the critic CE
    # (reduces leakage) with the critic FROZEN in Step B -- this IS the envelope-theorem
    # profile gradient. A GRL would double-flip the sign here, so there is no grl option.
    cross_fit: bool = True              # report cross-fitted held-out critic estimate
    # primal-dual leakage budget (review 5.5): lambda_j <- [lambda_j + eta(I_j - eps_j)]_+
    dual_lr: float = 0.5
    lambda_init: float = 0.1
    lambda_max: float = 20.0
    warmup: int = 5                     # epochs before the dual variables start moving


@dataclass
class AlignConfig:
    """Reference-prior marginal alignment (review section 4 / P0-5)."""
    enabled: bool = True
    weight: float = 1.0
    distance: str = "sliced_wasserstein"   # 'sliced_wasserstein' | 'energy' | 'gauss_w2'
    n_projections: int = 64             # for sliced Wasserstein
    reference_prior: str = "uniform"    # pi*(y): 'uniform' | 'source_marginal'


@dataclass
class DisentangleConfig:
    """Task/nuisance latent separation (review 5.6)."""
    enabled: bool = True
    rho: float = 0.1                    # weight on L_disentangle
    eta_zn_y: float = 0.1               # penalise extra I(Z_n;Y|D)
    kappa_zn_d: float = 0.1             # encourage I(Z_n;D) (nuisance carries domain)
    method: str = "hsic"               # 'hsic' | 'cross_cov'


@dataclass
class SSLConfig:
    """Self-supervised / reconstruction aux to avoid z_c collapse (review 5.2/5.6)."""
    enabled: bool = True
    weight: float = 0.1
    masked_recon: bool = True
    vicreg: bool = True                 # variance/covariance regularisation on z_c
    mask_ratio: float = 0.3


@dataclass
class TTAConfig:
    """Selective class-conditional probabilistic TTA (review section 6/7)."""
    transform: str = "diag_affine"      # 'diag_affine' | 'lowrank_affine' | 'full_affine'
    lowrank: int = 4
    trust_region: float = 1.0           # tau: ||A - I||_F^2 penalty
    trust_region_b: float = 1.0         # tau_b: ||b||^2 penalty
    logdet_weight: float = 1.0          # n*log|det A| anti-collapse term
    # Dirichlet-anchor concentrations on pi_T (a KL-toward-pi_S MAP regulariser): the
    # M-step prior uses pseudo-counts (dirichlet + prior_kl) * pi_S, so `prior_kl` is the
    # genuine pull toward the source prior and `dirichlet` the anti-collapse floor.
    prior_kl: float = 1.0
    dirichlet: float = 5.0
    em_iters: int = 20
    em_lr: float = 5e-2
    online_ema: float = 0.9             # for the streaming prior-only variant
    min_target: int = 16                # below this -> identity fallback
    min_effective_classes: int = 2      # single-class target -> identity fallback
    # cross-fitted held-out change-of-variable NLL improvement required to adapt; otherwise
    # identity rollback (review P0-4: same data must not both fit and judge the transform).
    min_heldout_evidence: float = 0.0


@dataclass
class GateConfig:
    """Source-only learned safety gate (review section 7)."""
    enabled: bool = True
    model: str = "logistic"             # 'logistic' | 'gbt'
    harm_delta: float = 0.0             # G < -delta counts as harm
    risk_threshold: float = 0.5         # adapt only if P(harm|g) < threshold
    min_evidence: float = 0.0           # require density NLL improvement >= this


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 5
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 0
    drop_last: bool = False             # review code-issue: do NOT drop rare domain-class cells


@dataclass
class H2Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    cmi: CMIConfig = field(default_factory=CMIConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    disentangle: DisentangleConfig = field(default_factory=DisentangleConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    tta: TTAConfig = field(default_factory=TTAConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    n_classes: int = 2

    def small(self) -> "H2Config":
        """Shrink everything for the smoke test / CPU runs (mutates and returns self)."""
        self.train.epochs = 4
        self.cmi.critic_inner = 1
        self.tta.em_iters = 8
        return self


def core_config(cfg: H2Config) -> H2Config:
    """The MINIMAL trustworthy core (review section 4): encoder + p_phi(z|y) +
    hierarchical CMI + offline diagonal TTA. Everything whose optimisation direction or
    evaluation protocol still needs work is OFF until validated piece by piece:

      disentanglement   (min-min adversary surrogate needs alternating Step A/B)
      SSL reconstruction(z_c reconstructing raw EEG fights the CMI objective)
      source canonicalizer (absorbable by the fusion layer; not yet identifiable)
      safety gate       (not yet a truly nested inner-LOSO; pseudo-targets saw training)
      online transform  (deferred; only prior-only streaming is causal today)
      reference alignment(needs domain-class balanced batches / LOO reference)

    Mutates and returns ``cfg``.
    """
    cfg.encoder.z_c_dim = 16
    cfg.encoder.z_n_dim = 8
    cfg.encoder.use_temporal = True
    cfg.encoder.use_spd = True
    cfg.encoder.use_graph = False
    cfg.encoder.canonicalizer = False
    cfg.density.n_components = 1
    cfg.density.cov_rank = 2
    cfg.density.df = 8.0
    cfg.align.enabled = False
    cfg.disentangle.enabled = False
    cfg.ssl.enabled = False
    cfg.gate.enabled = False
    cfg.tta.transform = "diag_affine"
    cfg.tta.em_iters = 20
    cfg.cmi.critic_inner = 5
    return cfg
