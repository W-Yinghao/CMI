"""Tests for scripts/collect_cigl_evidence_tables.py (Phase 4A; no training, no GPU)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.collect_cigl_evidence_tables as C  # noqa: E402


def test_runs_dry_without_generated_json(monkeypatch, tmp_path):
    """Dry run tolerates missing summary JSON and still emits the claims-scope table."""
    out = tmp_path / "tables"
    monkeypatch.setattr(C, "CONFIRMATIONS", [
        ("bnci2014_confirm", tmp_path / "missing_2014.json", "note"),
        ("bnci2015_confirm", tmp_path / "missing_2015.json", "note"),
    ])
    monkeypatch.setattr(sys, "argv", ["prog", "--dry_run", "--out_dir", str(out)])
    assert C.main() == 0
    assert (out / "table_claims_scope.md").exists()              # claims scope always written
    # no per-fold tables when JSON missing
    assert not (out / "table_bnci2014_confirm.md").exists()


def test_output_dir_created_in_temp(tmp_path, monkeypatch):
    out = tmp_path / "nested" / "tables"
    monkeypatch.setattr(C, "CONFIRMATIONS", [])
    monkeypatch.setattr(sys, "argv", ["prog", "--out_dir", str(out)])
    assert C.main() == 0
    assert out.is_dir() and (out / "table_claims_scope.md").exists()


def test_no_gpu_or_training_imports():
    """The collector must not IMPORT torch / moabb / training modules (AST-checked; stdlib only)."""
    import ast
    src = (REPO / "scripts" / "collect_cigl_evidence_tables.py").read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"torch", "moabb", "mne", "numpy"}                  # heavy / training / GPU-adjacent deps
    assert not (imported & banned), f"unexpected heavy imports: {imported & banned}"
    # no submodule imports of the training/model code either
    assert "cmi.train" not in src and "cmi.models" not in src.replace("cmi.models / training", "")
    assert not any(s in src for s in ("from cmi.train", "import cmi.train", "from cmi.models", "import cmi.models"))


def test_claims_scope_contains_required_phrases(tmp_path, monkeypatch):
    out = tmp_path / "t"
    monkeypatch.setattr(C, "CONFIRMATIONS", [])
    monkeypatch.setattr(sys, "argv", ["prog", "--dry_run", "--out_dir", str(out)])
    C.main()
    txt = (out / "table_claims_scope.md").read_text().lower()
    assert "no edge-cmi" in txt
    assert "no sota" in txt
    assert "posterior-kl proxy" in txt
    assert "source-only" in txt
    assert "elimination" in txt          # explicitly disclaims leakage elimination


def test_per_fold_table_built_from_fake_summary(tmp_path, monkeypatch):
    """When a summary JSON exists, a per-fold CSV+MD is emitted with the verdict footer."""
    import json
    j = tmp_path / "BNCI2015_001_dgcnn_gn_2nd_dataset_summary.json"
    summary = {
        "per_fold": {
            "0": {"heldout_subject": "1", "erm_fixed": {"source_probe_bacc": 0.69, "graph_kl_mean": 1.05},
                  "graph_node_010": {"source_probe_bacc": 0.689, "graph_kl_mean": 0.24,
                                     "graph_clears_seeds": 3, "node_clears_seeds": 3},
                  "flags": {"source_drop_vs_erm": 0.003, "graph_reduction": 0.77, "node_reduction": 0.57,
                            "source_retained": True, "target_guardrail": True, "fold_pass": True}},
        },
        "second_dataset_confirmation": {"source_only_confirmed": True, "target_guardrail_pass": True,
                                        "confirmed_with_target_guardrail": True, "decision": "A"},
    }
    j.write_text(json.dumps(summary))
    out = tmp_path / "tables"
    monkeypatch.setattr(C, "CONFIRMATIONS", [("bnci2015_confirm", j, "BNCI2015_001 note")])
    monkeypatch.setattr(sys, "argv", ["prog", "--out_dir", str(out)])
    assert C.main() == 0
    md = (out / "table_bnci2015_confirm.md").read_text()
    assert "confirmed_with_target_guardrail=True" in md and "decision=A" in md
    assert (out / "table_bnci2015_confirm.csv").exists()
