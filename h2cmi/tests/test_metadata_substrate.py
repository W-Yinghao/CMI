"""B2a metadata substrate: classifiers, the frozen rule table, missingness->identity, the causal
direction (metadata first), and the two anti-proxy properties that prevent it being a renamed
scenario lookup."""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

from h2cmi.data.metadata_substrate import (MetadataDelta, classify_geometry, classify_prevalence,
                                           metadata_to_operator, finalize, sample_episode,
                                           CANONICAL_CC, GEOMETRY_LEVELS, PREVALENCE_LEVELS)


def _d(**kw):
    base = dict(device_same=True, reference_same=True, montage_same=True, channel_layout_same=True,
                sampling_filter_same=True, cohort_same=True, sampling_protocol_same=True)
    base.update(kw)
    return finalize(MetadataDelta(**base))


def test_geometry_classifier():
    assert classify_geometry(_d()) == "NONE"
    assert classify_geometry(_d(device_same=False)) == "DIAG_COMPATIBLE"      # gain-like
    assert classify_geometry(_d(montage_same=False)) == "DIAG_COMPATIBLE"
    assert classify_geometry(_d(reference_same=False)) == "UNSUPPORTED"       # off-diagonal
    assert classify_geometry(_d(channel_layout_same=False)) == "UNSUPPORTED"
    assert classify_geometry(_d(device_same=None)) == "UNKNOWN"


def test_prevalence_classifier():
    assert classify_prevalence(_d()) == "SAME"
    assert classify_prevalence(_d(cohort_same=False)) == "DIFFERENT"
    assert classify_prevalence(_d(sampling_protocol_same=False)) == "DIFFERENT"
    assert classify_prevalence(_d(cohort_same=None)) == "UNKNOWN"


def test_frozen_rule_table():
    assert metadata_to_operator(_d()) == "identity"                                   # NONE
    assert metadata_to_operator(_d(device_same=False)) == "pooled_empirical_diag"     # DIAG/SAME
    assert metadata_to_operator(_d(device_same=False, cohort_same=False)) == CANONICAL_CC  # DIAG/DIFFERENT
    assert metadata_to_operator(_d(device_same=False, cohort_same=None)) == "identity"     # DIAG/UNKNOWN
    assert metadata_to_operator(_d(reference_same=False, cohort_same=False)) == "identity"  # UNSUPPORTED -> identity
    assert metadata_to_operator(_d(device_same=None)) == "identity"                        # UNKNOWN geom
    # pure prior risk (no geometry change) must map to identity
    assert metadata_to_operator(_d(cohort_same=False)) == "identity"


def test_unsupported_never_adapts():
    # a detected reference/layout change must NOT trigger diagonal adaptation while SPD/rotation frozen
    for prev in (True, False, None):
        assert metadata_to_operator(_d(reference_same=False, cohort_same=prev)) == "identity"


def _episodes(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return [sample_episode(rng) for _ in range(n)]


def test_causal_metadata_first_and_residual_drift():
    eps = _episodes()
    # NONE-geometry episodes carry NO geometry knob but DO carry residual session drift
    none_eps = [e for e in eps if e.delta.geometry_compatibility == "NONE"]
    assert none_eps and all(e.scenario.target_gain == 0.0 and e.scenario.target_cov == 0.0 for e in none_eps)
    assert all(e.scenario.target_noise_delta > 0 for e in eps)
    # SAME-prevalence episodes never realise a prior shift
    same = [e for e in eps if e.delta.prevalence_risk == "SAME"]
    assert same and all(e.scenario.target_prior == 0.0 for e in same)


def test_anti_proxy_same_metadata_many_latents():
    eps = _episodes()
    by_meta = defaultdict(set)
    for e in eps:
        key = (e.delta.geometry_compatibility, e.delta.prevalence_risk)
        by_meta[key].add(e.latent_stratum)
    # the DIAG/DIFFERENT metadata tuple must map to MANY latent strata (magnitude is drawn)
    assert len(by_meta[("DIAG_COMPATIBLE", "DIFFERENT")]) >= 3


def test_anti_proxy_same_latent_many_metadata():
    eps = _episodes()
    by_latent = defaultdict(set)
    for e in eps:
        by_latent[e.latent_stratum].add(e.delta.geometry_compatibility)
    # a low-but-nonzero geometry stratum must be reachable from >= 2 distinct geometry tuples
    low = [s for s in by_latent if s.startswith("g=L")]
    assert low and any(len(by_latent[s]) >= 2 for s in low), "latent->metadata is a 1:1 lookup"


def test_missingness_present_and_routes_identity():
    eps = _episodes()
    unk_g = [e for e in eps if e.delta.geometry_compatibility == "UNKNOWN"]
    unk_p = [e for e in eps if e.delta.prevalence_risk == "UNKNOWN"]
    assert unk_g and unk_p                                            # frozen missingness occurs
    assert all(metadata_to_operator(e.delta) == "identity" for e in unk_g)


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_metadata_substrate PASSED")
