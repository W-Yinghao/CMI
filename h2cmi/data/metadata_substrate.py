"""Stage-B2a metadata substrate (review (a')): a two-axis, deployable, causally-correct metadata
layer for metadata-conditioned operator selection. The router sees ONLY the observable metadata;
the latent acquisition transform + class prior are generated FROM the metadata (never the reverse),
with calibrated imperfection so the metadata is NOT a renamed scenario lookup.

Axis 1  acquisition-geometry compatibility (from BIDS-EEG-recordable fields: device/amplifier,
        reference scheme, cap/montage/placement, channel set & layout, sampling-rate/filtering,
        site/session hierarchy) -> {NONE, DIAG_COMPATIBLE, UNSUPPORTED, UNKNOWN}.
Axis 2  prevalence-risk (from study-design fields available at deployment WITHOUT target labels:
        cohort/ascertainment, sampling protocol) -> {SAME, DIFFERENT, UNKNOWN}.

The frozen rule table maps (geometry, prevalence) -> a mechanism-level operator. UNSUPPORTED and
UNKNOWN always map to identity (do NOT run diagonal adaptation just because a montage/reference
change is *detected* -- the diagonal family cannot honestly express it while SPD/rotation stay
frozen). Only a DIAG-expressible geometry shift WITH a prevalence-risk signal triggers the class-
conditional operator; geometry-with-same-prevalence triggers pooled; pure prior risk -> identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from h2cmi.data.paired_simulator import ScenarioSpec

GEOMETRY_LEVELS = ("NONE", "DIAG_COMPATIBLE", "UNSUPPORTED", "UNKNOWN")
PREVALENCE_LEVELS = ("SAME", "DIFFERENT", "UNKNOWN")
CANONICAL_CC = "canonical_fixed_prior_class_conditional_diag"
OPERATORS = ("identity", "pooled_empirical_diag", CANONICAL_CC)

# magnitude ranges calibrated to the standard-grid shift scales (NOT tuned to B2a performance):
# standard cov/prior used knob=1.0; gain is multiplicative so it sits on a smaller scale.
_GAIN_MAX = 0.6           # DIAG_COMPATIBLE channel-gain magnitude
_COV_MAX = 1.0            # UNSUPPORTED off-diagonal mixing magnitude
_PRIOR_MAX = 1.0          # prevalence-DIFFERENT prior-shift magnitude
_RESIDUAL_DRIFT = 0.05    # session-level residual present even with NO metadata change


@dataclass
class MetadataDelta:
    """Observable source->target relation. Raw fields are deployable acquisition/study-design
    descriptors; the two axes are DERIVED by the frozen classifiers (stored for audit). None = a
    field is missing/unrecorded."""
    device_same: bool | None
    reference_same: bool | None
    montage_same: bool | None
    channel_layout_same: bool | None
    sampling_filter_same: bool | None
    cohort_same: bool | None
    sampling_protocol_same: bool | None
    geometry_compatibility: str = ""
    prevalence_risk: str = ""

    def axes(self) -> tuple[str, str]:
        return self.geometry_compatibility, self.prevalence_risk


def classify_geometry(d: MetadataDelta) -> str:
    """Deployable classifier: missing critical field -> UNKNOWN; no change -> NONE; reference or
    channel-layout change (off-diagonal/transport) -> UNSUPPORTED; otherwise a channel-wise
    gain/scale change the diagonal family can express -> DIAG_COMPATIBLE."""
    crit = (d.device_same, d.reference_same, d.montage_same, d.channel_layout_same, d.sampling_filter_same)
    if any(x is None for x in crit):
        return "UNKNOWN"
    if all(crit):
        return "NONE"
    if (d.reference_same is False) or (d.channel_layout_same is False):
        return "UNSUPPORTED"
    return "DIAG_COMPATIBLE"


def classify_prevalence(d: MetadataDelta) -> str:
    crit = (d.cohort_same, d.sampling_protocol_same)
    if any(x is None for x in crit):
        return "UNKNOWN"
    return "SAME" if all(crit) else "DIFFERENT"


# Frozen rule table (review (a')).
def metadata_to_operator(d: MetadataDelta) -> str:
    g, p = d.geometry_compatibility, d.prevalence_risk
    if g != "DIAG_COMPATIBLE":                       # NONE / UNSUPPORTED / UNKNOWN
        return "identity"
    if p == "SAME":
        return "pooled_empirical_diag"
    if p == "DIFFERENT":
        return CANONICAL_CC
    return "identity"                                # prevalence UNKNOWN


def finalize(d: MetadataDelta) -> MetadataDelta:
    d.geometry_compatibility = classify_geometry(d)
    d.prevalence_risk = classify_prevalence(d)
    return d


@dataclass
class B2aEpisode:
    delta: MetadataDelta             # observable (the ONLY router input)
    scenario: ScenarioSpec           # latent mechanism (hidden; fed to the simulator)
    latent_stratum: str              # analysis-only: binned realized effect magnitudes
    geom_magnitude: float
    prior_magnitude: float
    intended_geometry: str           # the geometry axis BEFORE missingness (analysis-only)
    meta: dict = field(default_factory=dict)


def _raw_for(intended_geom: str, intended_prev: str, rng) -> MetadataDelta:
    """Accurate raw fields consistent with the intended axes (tags are NOT flipped)."""
    g_same = dict(device_same=True, reference_same=True, montage_same=True,
                  channel_layout_same=True, sampling_filter_same=True)
    if intended_geom == "DIAG_COMPATIBLE":           # gain-like change: device/montage/filter, NOT ref/layout
        field_ = rng.choice(["device_same", "montage_same", "sampling_filter_same"])
        g_same[field_] = False
    elif intended_geom == "UNSUPPORTED":             # reference or channel-layout transport
        g_same[rng.choice(["reference_same", "channel_layout_same"])] = False
    p_same = dict(cohort_same=True, sampling_protocol_same=True)
    if intended_prev == "DIFFERENT":
        p_same[rng.choice(["cohort_same", "sampling_protocol_same"])] = False
    return MetadataDelta(**g_same, **p_same)


def _bin(x, edges=(1e-6, 0.2, 0.5)):
    return "Z" if x < edges[0] else "L" if x < edges[1] else "M" if x < edges[2] else "H"


def sample_episode(rng, *, geom_probs=(0.30, 0.30, 0.40), prev_probs=(0.55, 0.45),
                   missing_rate=0.12) -> B2aEpisode:
    """Causal direction: observable metadata FIRST -> latent acquisition transform + prior SECOND.
    Imperfection: (i) magnitude is DRAWN per episode (same metadata -> many latent realizations,
    sometimes ~0 effect); (ii) a residual session drift is always present; (iii) ascertainment only
    changes the prior-shift DISTRIBUTION, not the realised prior; (iv) frozen missingness -> UNKNOWN.
    The metadata->latent overlap at small magnitudes makes a given latent stratum reachable from
    multiple metadata tuples (anti-proxy)."""
    intended_geom = rng.choice(["NONE", "DIAG_COMPATIBLE", "UNSUPPORTED"], p=list(geom_probs))
    intended_prev = rng.choice(["SAME", "DIFFERENT"], p=list(prev_probs))
    d = _raw_for(intended_geom, intended_prev, rng)
    # frozen missingness (independent per axis) -> UNKNOWN
    if rng.random() < missing_rate:
        d.device_same = None
    if rng.random() < missing_rate:
        d.cohort_same = None
    finalize(d)
    g, p = d.axes()

    scen = ScenarioSpec(name="b2a", target_noise_delta=_RESIDUAL_DRIFT)
    geom_mag = 0.0
    # latent geometry conditioned on the OBSERVED axis (UNKNOWN: an unrecorded change may still exist)
    eff_geom = intended_geom if g == "UNKNOWN" else g
    if eff_geom == "DIAG_COMPATIBLE":
        geom_mag = float(rng.uniform(0, _GAIN_MAX)); scen.target_gain = geom_mag
    elif eff_geom == "UNSUPPORTED":
        geom_mag = float(rng.uniform(0, _COV_MAX)); scen.target_cov = geom_mag
    prior_mag = 0.0
    # ascertainment only shifts the DISTRIBUTION of prior change (DIFFERENT -> often, not always)
    draw_prior = (p == "DIFFERENT") or (p == "UNKNOWN" and rng.random() < 0.4)
    if draw_prior:
        prior_mag = float(abs(rng.uniform(0, _PRIOR_MAX))); scen.target_prior = prior_mag
    stratum = f"g={_bin(geom_mag)}/p={_bin(prior_mag)}"
    return B2aEpisode(d, scen, stratum, geom_mag, prior_mag, intended_geom,
                      meta=dict(eff_geom=eff_geom, observed=(g, p)))


# Route alias: which deployable variant each metadata route maps to (canonical = the FROZEN choice).
ROUTE_VARIANT = {"pooled": "pooled_empirical_diag", "cc": "gen_oneshot_diag"}


def route_bank_episode(rng, route: str, bank: str) -> B2aEpisode:
    """Source-only calibration bank for the route-CONDITIONED single-action gate. Builds a
    metadata-ROUTE-POSITIVE episode (geometry DIAG_COMPATIBLE so g(Δm) != identity; no missingness)
    with the NET GEOMETRY EFFECT controlled by `bank`:
      route 'pooled' -> DIAG x SAME ;  route 'cc' -> DIAG x DIFFERENT.
      bank 'null'  -> target_gain = 0 (true null); the CC route still lets the prior vary per the
                      frozen prevalence mechanism (so the CC action cannot auto-pass on pure prior).
      bank 'power' -> target_gain ~ the frozen B2a gain distribution.
    Used ONLY for source-only route-null FPR / route-alternative retention / ROC ceiling -- never to
    pick a threshold from target data."""
    assert route in ROUTE_VARIANT and bank in ("null", "power")
    prev = "SAME" if route == "pooled" else "DIFFERENT"
    d = finalize(_raw_for("DIAG_COMPATIBLE", prev, rng))
    scen = ScenarioSpec(name=f"b2b_{route}_{bank}", target_noise_delta=_RESIDUAL_DRIFT)
    geom_mag = 0.0
    if bank == "power":
        geom_mag = float(rng.uniform(0, _GAIN_MAX)); scen.target_gain = geom_mag
    prior_mag = 0.0
    if route == "cc" and rng.random() < 0.7:           # frozen prevalence mechanism varies the prior
        prior_mag = float(rng.uniform(0, _PRIOR_MAX)); scen.target_prior = prior_mag
    return B2aEpisode(d, scen, f"route={route}/bank={bank}", geom_mag, prior_mag, "DIAG_COMPATIBLE",
                      meta=dict(route=route, bank=bank, variant=ROUTE_VARIANT[route]))
