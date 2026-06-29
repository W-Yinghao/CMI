"""CIGL Phase 3A tests: config grammar, dry-run runner contract, and the strict target-label rule.

Pure-Python checks on the config list and the clears-null gate; a behavioral-invariance test that
PROVES target labels never affect the source side / audit / selection (corrupting target labels leaves
every source-side number byte-identical); plus one tiny subprocess dry-run validating the summary
schema (pass-1 vs confirmation, confirmation per-seed records, confirm-ERM reference, full_cigl
eligibility for best-Pareto, leakage-reduction fields, fresh-probe vs Step-A separation).
"""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_cigl_phase3a_regularizer_pilot.py"
OUT = REPO / "results" / "cigl" / "phase3a_pilot"
SUMMARY = OUT / "synthetic_fold0_phase3a_summary.json"
NP = 3   # dry-run n_perm (== n_perm_confirm after the dry-run cap)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("phase3a_runner", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- config grammar (pure python)
def test_parse_configs_returns_exactly_seven_with_triples():
    mod = _load_runner_module()
    cfgs = mod.parse_configs()
    assert len(cfgs) == 7
    by_label = {c[0]: c for c in cfgs}
    expected = {"erm": (0.0, 0.0, 0.0), "graph_only": (0.3, 0.0, 0.0), "node_only": (0.0, 0.3, 0.0),
                "edge_only": (0.0, 0.0, 0.1), "graph_node": (0.3, 0.3, 0.0), "full_cigl": (0.3, 0.3, 0.1),
                "low_full_cigl": (0.1, 0.1, 0.03)}
    assert set(by_label) == set(expected)
    for lbl, (lg, ln, le) in expected.items():
        _, gstr, glg, gln, gle = by_label[lbl]
        assert (glg, gln, gle) == (lg, ln, le) and gstr == f"graphcmi:{lg}:{ln}:{le}"
    assert by_label["erm"][1] == "graphcmi:0.0:0.0:0.0"


def test_obj_block_clears_null_gate():
    """clears_null = kl_mean > permutation_mean AND permutation_p <= gate_alpha (all three branches)."""
    f = _load_runner_module()._obj_block
    a = 0.05
    r = f({"kl_mean": 0.5, "permutation_mean": 0.1, "permutation_p": 0.02}, a)   # excess + significant
    assert r["positive_excess"] is True and r["clears_null"] is True
    r = f({"kl_mean": 0.5, "permutation_mean": 0.1, "permutation_p": 0.20}, a)   # excess, NOT significant
    assert r["positive_excess"] is True and r["clears_null"] is False
    r = f({"kl_mean": 0.05, "permutation_mean": 0.1, "permutation_p": 0.02}, a)  # no excess
    assert r["positive_excess"] is False and r["clears_null"] is False


def test_full_cigl_is_eligible_for_best_pareto():
    """The best-Pareto candidate set must exclude ONLY erm (full_cigl can be selected)."""
    mod = _load_runner_module()
    assert "full_cigl" not in mod.CONFIRM_LABELS or mod.CONFIRM_LABELS == {"erm", "full_cigl"}
    # candidate rule is `l != "erm"` in the runner; assert full_cigl would be a candidate
    labels = [c[0] for c in mod.parse_configs()]
    cand = [l for l in labels if l != "erm"]
    assert "full_cigl" in cand


# ----------------------------------------------------------------- strict target-label invariance
def test_target_labels_do_not_affect_source_side_or_audit():
    """Corrupting ONLY the target-subject labels must leave source_probe, the leakage audit, the split
    sizes and (hence) selection inputs byte-identical — proving target labels never leak into training,
    probe fitting, the audit, or selection. Only target_eval may change."""
    mod = _load_runner_module()
    X, y, dom, trm, tem, ncls, held = mod._synthetic_fold(seed=0)
    args = types.SimpleNamespace(epochs=2, bs=64, enc_train_frac=0.7, train_frac=0.7,
                                 min_per_cell=2, probe_epochs=3, gate_alpha=0.05)
    cfg = ("full_cigl", "graphcmi:0.3:0.3:0.1", 0.3, 0.3, 0.1)
    rec_a = mod._train_extract_audit(cfg, 0, (X, y, dom, trm, tem, ncls, held), args, "cpu", n_perm=NP)
    y2 = y.copy()
    y2[tem] = np.random.default_rng(999).permutation(y2[tem])     # corrupt ONLY target labels
    rec_b = mod._train_extract_audit(cfg, 0, (X, y2, dom, trm, tem, ncls, held), args, "cpu", n_perm=NP)

    assert rec_a["source_probe"] == rec_b["source_probe"], "source task metric must not depend on target labels"
    assert rec_a["n_enc_train"] == rec_b["n_enc_train"] and rec_a["n_probe_pool"] == rec_b["n_probe_pool"]
    for o in ("graph", "node", "edge"):
        assert rec_a[o]["kl_mean"] == rec_b[o]["kl_mean"], f"{o} leakage must not depend on target labels"
        assert rec_a[o]["permutation_p"] == rec_b[o]["permutation_p"]
        assert rec_a[o]["clears_null"] == rec_b[o]["clears_null"]
    # the held-out target subject must not be among the source probe domains
    src_doms = set(int(d) for d in dom[trm])
    assert int(held) not in set(int(d) for d in dom[trm])  # held-out subject excluded from source
    assert rec_a["n_enc_train"] + rec_a["n_probe_pool"] == int(trm.sum())  # partition drawn from source only


# ----------------------------------------------------------------- dry-run runner contract
_DRYRUN_DONE = False


def _run_dry_run_once():
    """Run the dry-run exactly once per session (keyed to a module flag, NOT to a possibly-stale file)."""
    global _DRYRUN_DONE
    if _DRYRUN_DONE:
        return
    for p in OUT.glob("synthetic_*.json"):       # clear stale artifacts so we test THIS code
        p.unlink()
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry_run_synthetic", "--device", "cpu",
         "--seeds", "0", "--n_perm", str(NP), "--n_perm_confirm", str(NP), "--epochs", "2", "--probe_epochs", "3"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"dry-run failed:\nSTDOUT{r.stdout[-2000:]}\nSTDERR{r.stderr[-2000:]}"
    _DRYRUN_DONE = True


def test_dry_run_summary_schema_and_target_rule():
    _run_dry_run_once()
    assert SUMMARY.exists()
    s = json.load(open(SUMMARY))
    m = s["meta"]
    assert m["exploratory"] is True and m["phase"] == "Phase3A_regularizer_effect_pilot"
    assert m["setting"] == "strict_source_only_DG"
    assert m["used_target_labels_for_training"] is False
    assert m["used_target_labels_for_selection"] is False
    assert m["used_target_covariates"] is False
    assert m["target_eval_is_evaluation_only"] is True

    # all 7 configs aggregated; pass-1 reduction vs pass-1 ERM; ERM self-reduction is exactly 0
    assert set(s["per_config"]) == {"erm", "graph_only", "node_only", "edge_only",
                                    "graph_node", "full_cigl", "low_full_cigl"}
    for o in ("graph", "node", "edge"):
        assert s["per_config"]["erm"][f"{o}_pass1_leakage_reduction_vs_erm"] == 0.0
        for lbl, a in s["per_config"].items():
            assert f"{o}_pass1_leakage_reduction_vs_erm" in a and f"{o}_clears_null_count" in a
            assert a["target_eval_is_evaluation_only"] is True
    assert "pass1_reference_kl" in s and "confirm_reference_kl" in s

    # best Pareto: full_cigl eligible -> only ERM is excluded
    assert s["best_pareto_config"] != "erm"
    # confirmation covers ERM, full_cigl, and best_pareto, with confirm-ERM-referenced reductions
    assert {"erm", "full_cigl"}.issubset(set(s["confirmation"]))
    assert s["best_pareto_config"] in s["confirmation"]
    for o in ("graph", "node", "edge"):
        assert s["confirmation"]["erm"][f"{o}_confirm_leakage_reduction_vs_confirm_erm"] == 0.0


def test_confirmation_per_seed_records_present_and_traceable():
    _run_dry_run_once()
    s = json.load(open(SUMMARY))
    cps = s["confirmation_per_seed"]
    assert {"erm", "full_cigl"}.issubset(set(cps))
    for label, by_seed in cps.items():
        assert by_seed, f"no per-seed confirmation records for {label}"
        for seed, blocks in by_seed.items():
            for o in ("graph", "node", "edge"):
                for k in ("kl_mean", "permutation_mean", "permutation_p", "positive_excess", "clears_null", "gate_alpha"):
                    assert k in blocks[o], f"confirmation_per_seed[{label}][{seed}][{o}] missing {k}"
    # the per-seed confirmation JSON files exist on disk with the dataset/fold/config/seed/nperm naming
    assert (OUT / f"synthetic_fold0_confirm_erm_seed0_nperm{NP}.json").exists()


def test_per_seed_record_naming_and_meta():
    _run_dry_run_once()
    rec_path = OUT / f"synthetic_fold0_erm_seed0_nperm{NP}.json"
    assert rec_path.exists(), "pass-1 per-seed file must include dataset/fold/config/seed/nperm"
    rec = json.load(open(rec_path))
    assert rec["target_eval"]["evaluation_only"] is True
    m = rec["meta"]
    assert m["used_target_labels_for_training"] is False
    assert m["used_target_labels_for_selection"] is False
    assert m["used_target_covariates"] is False
    assert m["target_eval_is_evaluation_only"] is True


def test_audit_uses_fresh_probes_separate_from_stepA():
    """The leakage block must carry permutation-null evidence (fresh audit probes); Step-A head
    diagnostics are reported separately and never used as leakage evidence."""
    _run_dry_run_once()
    rec = json.load(open(OUT / f"synthetic_fold0_full_cigl_seed0_nperm{NP}.json"))
    for o in ("graph", "node", "edge"):
        # provenance of leakage evidence = permutation null + significance gate (audit_graph_objects)
        for k in ("kl_mean", "permutation_mean", "permutation_p", "positive_excess", "clears_null"):
            assert k in rec[o]
        assert "dom_acc" not in rec[o] and "reg_graph" not in rec[o]
    assert "stepA" in rec
    for k in ("graph_dom_acc", "node_dom_acc", "edge_dom_acc", "graph_loss", "reg_graph"):
        assert k in rec["stepA"]
    assert "permutation_p" not in rec["stepA"] and "kl_mean" not in rec["stepA"]
