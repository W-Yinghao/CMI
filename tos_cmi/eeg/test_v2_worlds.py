"""V2 world-construction tests. Runnable as `python -m tos_cmi.eeg.test_v2_worlds` (prints PASS/FAIL) or via
pytest. Uses a controlled synthetic fixture with a clean linear task so the expected gate behaviours hold with
large margins. Covers: World A ceiling (oracle target gain but source-uncertifiable), World B unsafe reject,
World C removable-but-useless, target-labels-not-used-by-gate, and oracle-marked-diagnostic.
"""
from __future__ import annotations
import numpy as np

from tos_cmi.eeg.semi_synthetic_real_latent import inject
from tos_cmi.eeg.run_v2_certificate import eval_v2
from tos_cmi.eeg.v2_worlds import (FACTORIES, DEPLOYABLE, PRINCIPLED, DIAGNOSTIC,
                                   oracle_nuisance_eraser_factory)
from tos_cmi.eeg.source_ood_benefit_gate import gate_action


def _fixture(n_subj=12, per=100, d=8, mu=1.6, seed=0):
    """Clean linear task on dim0 with subject structure; a separate target subject."""
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), per)
    ys = rng.integers(0, 2, n_subj * per)
    Zs = rng.standard_normal((n_subj * per, d))
    Zs[:, 0] += (2 * ys - 1) * mu
    Zs += (subj % 3)[:, None] * 0.2                      # mild per-subject offset (real subject nuisance)
    nt = 120
    yt = rng.integers(0, 2, nt)
    Zt = rng.standard_normal((nt, d)); Zt[:, 0] += (2 * yt - 1) * mu
    return Zs, ys, subj, Zt, yt


def _signals(world, eraser_name, alpha=2.0, m=4, seed=0, phi=0.15, mu=1.6):
    Zs, ys, subj, Zt, yt = _fixture(seed=seed, mu=mu)
    inj = inject(world, Zs, ys, subj, Zt, yt, alpha=alpha, beta=1.0, phi=phi, seed=seed, m=m)
    F = oracle_nuisance_eraser_factory(m) if eraser_name in DIAGNOSTIC else FACTORIES[eraser_name]
    return inj, eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt, 2, F, seed, 8)


def test_z_relationships():
    Zs, ys, subj, Zt, yt = _fixture()
    a = inject("A", Zs, ys, subj, Zt, yt, alpha=1.0, phi=0.15, seed=0)
    b = inject("B", Zs, ys, subj, Zt, yt, alpha=1.0, seed=0)
    c = inject("C", Zs, ys, subj, Zt, yt, alpha=1.0, seed=0)
    ca = abs(np.corrcoef(a["z_src"], ys)[0, 1])
    assert 0.02 < ca < 0.6, "World A corr(z,y)=%.3f should be modest positive" % ca
    assert np.array_equal(a["z_tgt"], 1 - yt), "World A target z must be reversed (1-y)"
    assert np.array_equal(b["z_src"], ys) and np.array_equal(b["z_tgt"], yt), "World B z must equal y"
    assert abs(np.corrcoef(c["z_src"], ys)[0, 1]) < 0.15, "World C z must be ~independent of y"
    return "z-relationships A/B/C correct (A corr=%.3f, C~0)" % ca


def test_target_labels_not_used_by_gate():
    """Permuting the target labels used for SCORING must not change any SOURCE-only gate signal."""
    Zs, ys, subj, Zt, yt = _fixture()
    inj = inject("A", Zs, ys, subj, Zt, yt, alpha=2.0, phi=0.15, seed=0)
    F = FACTORIES["leace_baseline"]
    s1 = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt, 2, F, 0, 8)
    yt_perm = np.random.default_rng(9).permutation(yt)
    s2 = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt_perm, 2, F, 0, 8)
    for k in ("task_drop", "domain_gain"):
        assert abs(s1[k] - s2[k]) < 1e-9, "%s changed when target labels permuted (leak!)" % k
    assert np.allclose(s1["benefit"], s2["benefit"]), "source-LOSO benefit changed with target labels (leak!)"
    assert abs(s1["tgt_bacc_full"] - s2["tgt_bacc_full"]) > 1e-6, "sanity: target audit SHOULD change"
    return "gate signals invariant to target-label permutation (no leak); target audit does change"


def test_worldB_unsafe_reject():
    inj, s = _signals("B", "leace_baseline", alpha=2.0)
    assert s["task_drop"] > 0.05, "World B erasing z=y should collapse task (drop=%.3f)" % s["task_drop"]
    act = gate_action(s["task_drop"], (np.mean(s["benefit"]) if s["benefit"] else float("nan")))
    assert act == "REJECT", "World B gate should REJECT, got %s" % act
    return "World B unsafe: task-drop %.3f -> REJECT" % s["task_drop"]


def test_worldC_removable_but_useless():
    inj, s = _signals("C", "leace_baseline", alpha=2.0)
    assert s["domain_gain"] > 0.05, "World C should have high domain-gain (z erased), got %.3f" % s["domain_gain"]
    assert abs(s["task_drop"]) < 0.05, "World C erasing z should NOT hurt task (drop=%.3f)" % s["task_drop"]
    blcb = np.mean(s["benefit"]) if s["benefit"] else 0.0
    assert blcb <= 0.05, "World C should show ~no source-LOSO benefit (%.3f)" % blcb
    return "World C useless: domain-gain %.3f, task-drop %.3f, benefit~%.3f -> not accept" % (
        s["domain_gain"], s["task_drop"], blcb)


def test_worldA_ceiling():
    """Oracle nuisance removal improves the target, but the deployable gate does not ACCEPT (source-uncertifiable).
    Uses the regime where the head actually USES the nuisance (weaker real task, more aligned subjects, stronger
    nuisance) -- otherwise a too-clean fixture makes the nuisance inert (as on very strongly separable data)."""
    _, orc = _signals("A", "oracle_nuisance_eraser_DIAGNOSTIC_ONLY", alpha=3.0, m=6, phi=0.5, mu=0.7)
    _, lea = _signals("A", "leace_baseline", alpha=3.0, m=6, phi=0.5, mu=0.7)
    orc_gain = orc["tgt_bacc_eras"] - orc["tgt_bacc_full"]
    assert orc_gain > 0.005, "World A oracle should improve target (gain=%.3f)" % orc_gain
    lea_blcb = np.mean(lea["benefit"]) if lea["benefit"] else float("nan")
    act = gate_action(lea["task_drop"], lea_blcb)
    assert act != "ACCEPT", "World A deployable gate must NOT accept (got %s)" % act
    return "World A ceiling: oracle target gain +%.3f, deployable gate=%s (not ACCEPT)" % (orc_gain, act)


def test_config_parses_and_has_required_keys():
    """P0 gate: the frozen config MUST be valid YAML with all required keys and the locked values."""
    import yaml
    cfg = yaml.safe_load(open("tos_cmi/eeg/configs/v2_certificate_fixed.yaml"))
    required = ["goal", "datasets", "backbones", "seeds", "safety_reject_task_drop_ucb", "benefit_accept_lcb",
               "domain_gain_role", "target_usage", "world_A", "world_B", "world_C", "interventions"]
    missing = [k for k in required if k not in cfg]
    assert not missing, "missing config keys: %s" % missing
    assert cfg["goal"] == "source_only_acceptance_ceiling", cfg["goal"]
    assert cfg["safety_reject_task_drop_ucb"] == 0.02 and cfg["benefit_accept_lcb"] == 0.01
    assert cfg["domain_gain_role"] == "diagnostic_only" and cfg["target_usage"] == "audit_only"
    assert cfg["world_A"]["acceptance_expected"] is False
    assert cfg["world_B"]["acceptance_expected"] is False
    assert cfg["world_C"]["acceptance_expected"] is False
    assert "oracle_nuisance_eraser_DIAGNOSTIC_ONLY" in cfg["interventions"]
    assert "cc_leace_predicted_route_deployable" not in cfg["interventions"]
    return "config parses (%d keys); goal=ceiling; thresholds 0.02/0.01; oracle in interventions" % len(cfg)


def test_stage2_config_parses_and_scoped():
    """Stage-2 scoped config: valid YAML, required keys, World A=EEGNet-only / B,C=both, thresholds frozen,
    stop_conditions all set, oracle in interventions."""
    import yaml
    from pathlib import Path
    p = Path("tos_cmi/eeg/configs/v2_stage2_scoped.yaml")
    assert b"\r" not in p.read_bytes()
    cfg = yaml.safe_load(p.read_text())
    req = ["stage2_goal", "world_A", "world_B", "world_C", "datasets", "seeds", "source_subject_counts",
           "folds", "alpha_grid", "thresholds", "interventions", "stop_conditions"]
    assert not [k for k in req if k not in cfg], [k for k in req if k not in cfg]
    assert cfg["world_A"]["include_backbones"] == ["EEGNet"]
    assert "TSMNet" in cfg["world_A"]["exclude_backbones"]
    assert sorted(cfg["world_B"]["include_backbones"]) == ["EEGNet", "TSMNet"]
    assert sorted(cfg["world_C"]["include_backbones"]) == ["EEGNet", "TSMNet"]
    assert cfg["thresholds"]["safety_reject_task_drop_ucb"] == 0.02
    assert cfg["thresholds"]["benefit_accept_lcb"] == 0.01
    assert cfg["thresholds"]["target_usage"] == "audit_only"
    assert "oracle_nuisance_eraser_DIAGNOSTIC_ONLY" in cfg["interventions"]
    assert all(cfg["stop_conditions"].values())
    return "stage2 config parses; World A=EEGNet-only, B/C=both, thresholds frozen, 7 stop_conditions set"


def test_oracle_marked_diagnostic():
    assert "oracle_nuisance_eraser_DIAGNOSTIC_ONLY" in DIAGNOSTIC
    assert "oracle_nuisance_eraser_DIAGNOSTIC_ONLY" not in DEPLOYABLE
    assert "oracle_nuisance_eraser_DIAGNOSTIC_ONLY" not in PRINCIPLED
    E = oracle_nuisance_eraser_factory(4)(np.zeros((5, 10)), None, None, 2, 0)
    X = np.ones((5, 10)); out = E(X)
    assert np.all(out[:, -4:] == 0) and np.all(out[:, :6] == 1), "oracle must zero exactly the last m dims"
    return "oracle eraser is DIAGNOSTIC (not deployable) and zeros the injected block"


TESTS = [test_config_parses_and_has_required_keys, test_stage2_config_parses_and_scoped,
         test_z_relationships, test_target_labels_not_used_by_gate, test_worldB_unsafe_reject,
         test_worldC_removable_but_useless, test_worldA_ceiling, test_oracle_marked_diagnostic]


def main():
    ok = 0
    for t in TESTS:
        try:
            msg = t(); ok += 1; print("  PASS %-40s %s" % (t.__name__, msg))
        except AssertionError as e:
            print("  FAIL %-40s %s" % (t.__name__, e))
        except Exception as e:
            print("  ERROR %-40s %r" % (t.__name__, e))
    print("\n%d/%d V2 world tests passed" % (ok, len(TESTS)))
    print("V2_TESTS_DONE" if ok == len(TESTS) else "V2_TESTS_FAILED")


if __name__ == "__main__":
    main()
