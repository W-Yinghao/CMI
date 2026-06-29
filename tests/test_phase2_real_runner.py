"""CIGL Phase 2-real tests: the dry-run path of scripts/run_cigl_phase2_real_probe.py (CPU, no data).

Drives the actual script via subprocess (same interpreter) so the full dry-run pipeline — synthetic
features -> support-aware split -> audit -> map stability -> JSON — is exercised end to end, and the
exploratory / strict-source-only meta flags are checked. No EEG data, no moabb, no braindecode needed.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_cigl_phase2_real_probe.py"
SUMMARY = REPO / "results" / "cigl" / "phase2_real" / "synthetic_summary.json"


def test_dry_run_emits_valid_exploratory_summary():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry_run_synthetic", "--device", "cpu",
         "--seeds", "0", "1", "--n_perm", "3", "--map_stability_perms", "50"],
        cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"dry-run failed:\nSTDOUT{r.stdout[-1500:]}\nSTDERR{r.stderr[-1500:]}"
    assert SUMMARY.exists(), "synthetic_summary.json not written"
    s = json.load(open(SUMMARY))

    # strict source-only / exploratory contract
    m = s["meta"]
    assert m["exploratory"] is True
    assert m["used_target_labels"] is False and m["used_target_covariates"] is False
    assert m["setting"] == "strict_source_only_DG"
    assert m["dataset"] == "synthetic"
    assert "commit_hash" in m and "config_hash" in m

    # per-seed leakage rows present for both seeds, all three objects, with null + p
    assert len(s["per_seed"]) == 2
    for row in s["per_seed"]:
        for obj in ("graph", "node", "edge"):
            for k in ("kl_mean", "permutation_mean", "permutation_p"):
                assert k in row[obj], f"missing per_seed.{obj}.{k}"

    # map stability computed across the 2 seeds
    assert "map_stability" in s and s["map_stability"], "map_stability missing for >=2 seeds"
    for obj in ("node", "edge"):
        assert "stability" in s["map_stability"][obj] and "null" in s["map_stability"][obj]


def test_dry_run_per_seed_records_exist_and_are_exploratory():
    # the previous test already ran the dry-run; check a per-seed record file
    rec = REPO / "results" / "cigl" / "phase2_real" / "synthetic_seed0.json"
    if not rec.exists():                       # ensure independence if run in isolation
        subprocess.run([sys.executable, str(SCRIPT), "--dry_run_synthetic", "--device", "cpu",
                        "--seeds", "0", "--n_perm", "3"], cwd=str(REPO), check=True, timeout=600)
    d = json.load(open(rec))
    assert d["meta"]["exploratory"] is True
    assert d["meta"]["setting"] == "strict_source_only_DG"
    assert "probe_split_diagnostics" in d
    assert len(d["node"]["node_leakage_map"]) > 0
    assert "edge_leakage_top_k" in d["edge"]
