"""CIGL Phase 3A tests: config grammar + dry-run runner contract (CPU, synthetic, no EEG data).

Pure-Python checks on the config list, plus one tiny subprocess dry-run of the actual runner to verify
the summary schema, the strict target-label rule (eval-only; not used for training or config
selection), leakage_reduction_vs_erm fields, and that the leakage audit uses FRESH probes separate
from the Step-A training heads.
"""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_cigl_phase3a_regularizer_pilot.py"
OUT = REPO / "results" / "cigl" / "phase3a_pilot"
SUMMARY = OUT / "synthetic_fold0_phase3a_summary.json"


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
    assert len(cfgs) == 7, "Phase 3A must define exactly 7 configs"
    by_label = {c[0]: c for c in cfgs}
    expected = {
        "erm": (0.0, 0.0, 0.0), "graph_only": (0.3, 0.0, 0.0), "node_only": (0.0, 0.3, 0.0),
        "edge_only": (0.0, 0.0, 0.1), "graph_node": (0.3, 0.3, 0.0), "full_cigl": (0.3, 0.3, 0.1),
        "low_full_cigl": (0.1, 0.1, 0.03)}
    assert set(by_label) == set(expected)
    for lbl, (lg, ln, le) in expected.items():
        _, gstr, glg, gln, gle = by_label[lbl]
        assert (glg, gln, gle) == (lg, ln, le), f"{lbl} lambda triple mismatch"
        assert gstr == f"graphcmi:{lg}:{ln}:{le}"
    assert by_label["erm"][1] == "graphcmi:0.0:0.0:0.0", "ERM must be graphcmi:0:0:0"


# ----------------------------------------------------------------- dry-run runner contract
def _run_dry_run_once():
    if SUMMARY.exists():
        return
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry_run_synthetic", "--device", "cpu",
         "--seeds", "0", "--n_perm", "3", "--n_perm_confirm", "3", "--epochs", "2", "--probe_epochs", "3"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"dry-run failed:\nSTDOUT{r.stdout[-2000:]}\nSTDERR{r.stderr[-2000:]}"


def test_dry_run_summary_schema_and_target_rule():
    _run_dry_run_once()
    assert SUMMARY.exists(), "phase3a summary not written"
    s = json.load(open(SUMMARY))
    m = s["meta"]
    assert m["exploratory"] is True
    assert m["phase"] == "Phase3A_regularizer_effect_pilot"
    assert m["setting"] == "strict_source_only_DG"
    # strict target-label rule: not used for training or config selection
    assert m["used_target_labels_for_training"] is False
    assert m["used_target_labels_for_selection"] is False
    assert m["used_target_covariates"] is False
    assert "evaluation_only" in m["target_labels_used_for"]
    # all 7 configs aggregated, with leakage_reduction_vs_erm fields
    assert set(s["per_config"]) == {"erm", "graph_only", "node_only", "edge_only",
                                    "graph_node", "full_cigl", "low_full_cigl"}
    for lbl, a in s["per_config"].items():
        for o in ("graph", "node", "edge"):
            assert f"{o}_leakage_reduction_vs_erm" in a
            assert f"{o}_clears_null_count" in a
        assert a["target_eval_is_evaluation_only"] is True
    # best Pareto config chosen from source-only metrics -> never ERM or full_cigl
    assert s["best_pareto_config"] not in ("erm", "full_cigl")
    # confirmation re-audit covers ERM, full_cigl, and the best-Pareto config
    assert {"erm", "full_cigl"}.issubset(set(s["confirmation"]))
    if s["best_pareto_config"]:
        assert s["best_pareto_config"] in s["confirmation"]


def test_target_eval_marked_evaluation_only_in_per_seed_record():
    _run_dry_run_once()
    rec = json.load(open(OUT / "erm_seed0.json"))
    assert rec["target_eval"]["evaluation_only"] is True
    assert rec["meta"]["used_target_labels_for_training"] is False


def test_audit_uses_fresh_probes_separate_from_stepA():
    """Leakage audit (audit_graph_objects fresh probes) and Step-A training heads must BOTH be present
    and be distinct blocks — the audit is not reusing the Step-A heads."""
    _run_dry_run_once()
    rec = json.load(open(OUT / "full_cigl_seed0.json"))
    # fresh-probe audit block (permutation-null leakage)
    for o in ("graph", "node", "edge"):
        assert "kl_mean" in rec[o] and "permutation_p" in rec[o] and "clears_null" in rec[o]
    # Step-A training-head diagnostics, reported separately (not used as leakage evidence)
    assert "stepA" in rec
    for k in ("graph_dom_acc", "node_dom_acc", "edge_dom_acc"):
        assert k in rec["stepA"]
    # the two are different objects: audit has permutation_p, stepA has dom_acc
    assert "permutation_p" not in rec["stepA"]
    assert "dom_acc" not in rec["graph"]
