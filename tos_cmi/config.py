"""Config dataclasses for Task-Orthogonal Selective CMI (TOS-CMI).

Everything here is a plain dataclass so a run is fully described by one object and
results carry their config. Defaults are sized for the synthetic verifier; real-EEG
runs scale `subspace.max_dim` and the penalty weight up.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FisherConfig:
    """How the label Fisher F_Y and conditional-domain Fisher F_{D|Y} are estimated.

    Both are between-group scatter matrices (signal that moves the group means):
      F_Y     = between-class scatter of Z  over Y
      F_{D|Y} = E_y [ between-domain scatter of Z within class y ]   (class-conditional)
    `min_per_cell` drops (domain,class) cells with too few samples so a class with a
    single observed domain contributes 0 conditional-domain scatter rather than noise.
    """
    eta: float = 1e-3          # ridge on F_Y in the generalized eig (F_Y + eta I)
    min_per_cell: int = 2      # min samples in a (y) or (d,y) cell to count its mean
    center_global: bool = True # subtract the global mean before forming scatters
    dtype64: bool = True       # accumulate scatters / solve eig in float64


@dataclass
class SubspaceConfig:
    """Selection of the nuisance subspace P_N from the generalized spectrum.

    A direction is *nuisance-eligible* (domain-rich AND label-light) iff
      ratio_j  = dom_j / (lab_j + eta)        >= tau_ratio          (domain-rich)
      lab_j                                    <= eps_label * max_k lab_k   (label-light)
      dom_j                                    >= dom_floor * max_k dom_k   (not a null dir)
    where dom_j, lab_j are the F_{D|Y}- and F_Y-energies of the (Euclidean-unit)
    generalized eigenvector. We keep at most `max_dim` directions, highest ratio first.
    If NO direction qualifies the projector is the zero map => the method degrades to
    identity (this is the falsifiable "refuse to delete" behaviour, not a failure).
    """
    tau_ratio: float = 3.0     # domain/label energy ratio a direction must clear
    eps_label: float = 0.15    # max label-energy share (fraction of the most label-rich dir)
    dom_floor: float = 0.05    # min domain-energy share (reject near-null directions)
    max_dim: int = 8           # cap on nuisance-subspace dimension
    min_dim: int = 0           # if >0, require at least this many dirs or fall back to identity
    # Absolute domain-energy floor from a within-Y permutation null of D. A direction is
    # only "domain-rich" if its F_{D|Y} energy exceeds null_safety x (the largest domain
    # energy seen when D is shuffled within each class) -- i.e. above the sampling-noise
    # floor. This is what stops pure-noise / label-null directions (where B ~ eta) from
    # looking domain-rich. Set n_perm=0 to disable (not recommended).
    n_perm: int = 5
    null_safety: float = 1.25


@dataclass
class PenaltyConfig:
    """The selective leakage penalty lambda * I(P_N Z; D | Y)."""
    lam: float = 1.0           # penalty weight
    refresh_every: int = 5     # epochs between subspace re-estimations (Step-A cadence)
    inner_posterior_steps: int = 1  # Step-A critic updates per refresh batch
    prior_mode: str = "empirical"   # empirical | subject | effective (see cmi.methods.regularizers)


@dataclass
class TOSConfig:
    fisher: FisherConfig = field(default_factory=FisherConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    seed: int = 0
    device: str = "cpu"
