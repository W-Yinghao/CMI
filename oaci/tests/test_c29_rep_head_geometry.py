"""C29 Representation-Head Origin. Frozen C19 locked; logit=W·z+b so W·z=(logit−b) read-only; the decomposition
attributes the carrier to the representation-projection mean (R2) when the offset is encoded in W·z, and to the
parameter head-bias (R1) when encoded in b; counterfactuals + residual run; taxonomy deterministic; report
forbids selector language. Synthetic logits only."""
from __future__ import annotations

import numpy as np

from oaci.rep_head_geometry import (artifact_loader, counterfactual_logits, logit_decomposition, report,
                                    representation_persistence, schema, source_target_residual, taxonomy)


def _synth(rep_drives=True, n_targets=9, n_per=8, n_samp=40, seed=0):
    rng = np.random.RandomState(seed); score_rows = []; cands = []
    for t in range(1, n_targets + 1):
        off = (t - n_targets / 2) / (n_targets / 2)
        for k in range(n_per):
            good = k % 2 == 0; sc = off + (0.5 if good else -0.5) + rng.randn() * 0.15; mh = f"{t:02d}{k:03d}"
            score_rows.append({"mode": "in_regime", "regime": "S0", "seed": 0, "target": t, "level": 0,
                               "model_hash": mh, "score": sc, "label": 1 if good else 0})
            proj = rng.randn(n_samp, 4) * 0.5; b = rng.randn(4) * 0.1
            if rep_drives:
                proj[:, 0] += off * 2.0                       # representation projection encodes offset
            else:
                b[0] += off * 2.0                             # parameter head-bias encodes offset
            L = proj + b                                      # logit = W·z + b
            src_feats = {f"bias_c{c}": float(rng.randn()) for c in range(4)}
            cands.append({"seed": 0, "target": t, "level": 0, "model_hash": mh, "L": L, "b": b,
                          "weight_norms": np.ones(4), "src_feats": src_feats})
    raw, oracle, _, _ = artifact_loader.raw_oracle(score_rows, "in_regime")
    return cands, score_rows, raw, oracle


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_wz_is_logit_minus_b():
    c = {"L": np.arange(8.0).reshape(2, 4), "b": np.array([1.0, 2.0, 3.0, 4.0])}
    assert np.allclose(c["L"] - c["b"], np.array([[-1, -1, -1, -1], [3, 3, 3, 3]]))


def test_decomposition_R2_when_projection_encodes_offset():
    ca, sr, raw, oracle = _synth(rep_drives=True)
    d = logit_decomposition.logit_decomposition(ca, sr, "in_regime", raw, oracle)
    assert d["projection_mean_gauge"]["gap_closed"] > d["parameter_bias_gauge"]["gap_closed"]
    assert d["representation_projection_drives_offset"] is True and d["parameter_bias_drives_offset"] is False


def test_decomposition_R1_when_bias_encodes_offset():
    ca, sr, raw, oracle = _synth(rep_drives=False, seed=4)
    d = logit_decomposition.logit_decomposition(ca, sr, "in_regime", raw, oracle)
    assert d["parameter_bias_gauge"]["gap_closed"] > d["projection_mean_gauge"]["gap_closed"]
    assert d["parameter_bias_drives_offset"] is True


def test_counterfactuals_run():
    ca, sr, raw, oracle = _synth(rep_drives=True)
    cf = counterfactual_logits.counterfactuals(ca, sr, "in_regime", raw, oracle)
    assert set(cf["per_intervention"]) == set(schema.INTERVENTIONS)
    assert cf["per_intervention"]["raw"].get("destroys_recovery") is False


def test_residual_runs():
    ca, sr, raw, oracle = _synth(rep_drives=True)
    rd = source_target_residual.source_target_residual(ca, sr, "in_regime", raw, oracle)
    for k in ("target_projection_mean", "source_explained", "target_residual"):
        assert rd[k]["gap_closed"] is not None


def test_taxonomy_r2_and_r1():
    decomp = {"representation_projection_drives_offset": True, "parameter_bias_drives_offset": False}
    cf = {"destroyers": ["source_mean_centered_projection"]}
    rd = {"residual_over_source_explained": True, "source_explained": {"survives_permutation": False}, "residual_carries_offset": False}
    t = taxonomy.gauge_taxonomy(decomp, cf, rd, full_survives=True)
    assert t["primary_case"] == schema.R2
    assert schema.R3 in t["established"] and schema.R7 in t["established"]
    t2 = taxonomy.gauge_taxonomy({"representation_projection_drives_offset": False, "parameter_bias_drives_offset": True},
                                {"destroyers": []}, {"source_explained": {"survives_permutation": True}}, full_survives=True)
    assert t2["primary_case"] == schema.R1


def test_representation_persistence_offset_orthogonal():
    av = representation_persistence.availability()
    assert av["status"] == "NOT_NEEDED_OFFSET_ORTHOGONAL"
    assert "offset-relevant" in av["reason"] or "offset-orthogonal" in av["reason"]


def test_report_forbids_selector_language():
    for bad in ("a deployment-time selector", "representation selector achieved"):
        try:
            report._guard_forbidden(bad); raise AssertionError("guard failed to fire")
        except ValueError:
            pass
    report._guard_forbidden("NOT a deployment-time selector; diagnostic-only and identity-entangled.")
