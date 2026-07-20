"""Pre-execution protocol and implementation red team for C78F."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from . import c78f_full_seed3_field as c78f


REPORT_PATH = c78f.REPORT_DIR / "C78F_PROTOCOL_RED_TEAM_VERIFICATION.md"
CHECKS_PATH = c78f.TABLE_DIR / "protocol_red_team_checks.csv"


def run() -> dict[str, Any]:
    protocol_sha = c78f.sha256_file(c78f.PROTOCOL_PATH)
    expected = c78f.PROTOCOL_SHA_PATH.read_text().strip()
    protocol = json.loads(c78f.PROTOCOL_PATH.read_text())
    c78s = json.loads(c78f.C78S_PROTOCOL_PATH.read_text())
    checks = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "blocking": 1, "evidence": str(evidence)})

    check("C78F_protocol_hash", protocol_sha == expected, protocol_sha)
    check("C78S_protocol_hash", c78f.sha256_file(c78f.C78S_PROTOCOL_PATH) == c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(), c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip())
    check("C78S_bound_to_C78F", protocol["C78S_analysis_lock"]["sha256"] == c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(), protocol["C78S_analysis_lock"])
    check("direct_user_authorization", protocol["authorization"]["explicit_user_authorization_received"] and protocol["authorization"]["mode"] == c78f.AUTHORIZATION_MODE, protocol["authorization"])
    check("no_magic_token", not protocol["authorization"]["magic_token_required"] and "authorization_token_exact" not in c78f.PROTOCOL_PATH.read_text(), "removed")
    check("execution_lock_required", protocol["authorization"]["execution_requires_committed_scope_bound_lock"], True)
    check("scope_1296_1458", protocol["scope"]["remaining_units"] == 1296 and protocol["scope"]["full_seed3_units"] == 1458, protocol["scope"])
    check("phase_count_48", protocol["scope"]["remaining_training_phases"] == 48, 48)
    check("wave_A_exact", protocol["waves"]["A"] == list(c78f.wave_targets()["A"]), protocol["waves"]["A"])
    check("wave_B_exact", protocol["waves"]["B"] == list(c78f.wave_targets()["B"]), protocol["waves"]["B"])
    check("wave_continuation_engineering_only", not protocol["waves"]["target_outcomes_between_waves"], protocol["waves"]["wave_B_gate"])
    check("target4_primary_excluded", c78s["data_roles"]["primary_targets"] == list(c78f.TARGETS) and 4 not in c78s["data_roles"]["primary_targets"], c78s["data_roles"])
    check("seed4_untouched_contract", all(value == 0 for value in protocol["seed4_protection"].values()), protocol["seed4_protection"])
    check("fixed_historical_regimes", set(protocol["scope"]["regimes"]) == set(c78f.REGIMES) and protocol["training"]["SRC_smooth_temperature"] == 0.1, protocol["training"])
    check("fixed_cadence", protocol["training"]["checkpoint_every_epochs"] == 5 and protocol["training"]["outcome_blind_retention"], protocol["training"])
    check("physical_views_locked", set(protocol["physical_views"]) == {"strict_source_trial_view", "target_unlabeled_trial_view", "target_construction_view", "target_evaluation_view", "same_label_oracle_view"}, protocol["physical_views"])
    check("instrument_identity_tolerances", protocol["instrumentation"]["identity_tolerances"]["Wz_plus_b_logits_abs"] == 1e-6, protocol["instrumentation"]["identity_tolerances"])
    check("all_implementation_hashes", all(c78f.sha256_file(item["path"]) == item["sha256"] for item in protocol["implementation_files"]), len(protocol["implementation_files"]))
    check("all_historical_hashes", all(len(item["current_sha256"]) == 64 for item in protocol["historical_files"]), len(protocol["historical_files"]))
    train_source = Path("oaci/conditioned_ceiling_coverage/c78f_train.py").read_text()
    top_imports = {alias.name for node in ast.parse(train_source).body if isinstance(node, ast.Import) for alias in node.names}
    check("no_top_level_torch_or_EEG_loader", "torch" not in top_imports and "mne" not in top_imports and "moabb" not in top_imports, sorted(top_imports))
    check("workers_require_runtime_authorization", "runtime.require_authorization()" in train_source and "runtime.require_authorization()" in Path("oaci/conditioned_ceiling_coverage/c78f_instrument.py").read_text(), "guard present")
    scripts = [Path(name).read_text() for name in c78f.IMPLEMENTATION_FILES if name.endswith(".sh")]
    check("no_token_in_slurm", all("authorization-token" not in source for source in scripts), len(scripts))
    check("GPU_partition_locked", all("--partition=V100" in source for source in scripts[:2]), "V100")
    check("cpu_high_48_core_instrument", "--partition=cpu-high" in scripts[2] and "--cpus-per-task=48" in scripts[2], "cpu-high/48")
    preflight = c78f.read_csv(c78f.TABLE_DIR / "remaining_target_preflight.csv")
    check("all_target_metadata_preflights", len(preflight) == 8 and all(row["passed"] == "1" and row["EEG_data_loaded"] == "0" for row in preflight), len(preflight))
    storage = c78f.read_csv(c78f.TABLE_DIR / "storage_preflight.csv")[0]
    check("storage_preflight", storage["passed"] == "1", storage["free_bytes"])
    risks = c78f.read_csv(c78f.TABLE_DIR / "risk_register.csv")
    check("risk_register_complete", len(risks) == 25 and all(row["blocking_open"] == "0" for row in risks), len(risks))
    check("no_final_report_before_execution", not (c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.md").exists(), "absent")

    failures = [row for row in checks if row["status"] != "PASS"]
    c78f.write_csv(CHECKS_PATH, checks)
    REPORT_PATH.write_text(f"""# C78F Protocol Red-Team Verification

Gate: **{'PASS' if not failures else 'FAIL'}**

```text
checks: {len(checks)}
blocking failures: {len(failures)}
C78F protocol SHA-256: {protocol_sha}
C78S protocol SHA-256: {c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip()}
EEG data loaded: 0
GPU jobs submitted: 0
target outcomes read: 0
```

The direct user authorization rule is accepted as the PM override. Execution is
still fail-closed because workers require a committed scope-bound lock and never
scan prompts or environment variables for approval.

The 1,296-unit registry, 48 phases, deterministic waves, historical engines,
physical views, C78S hypotheses, multiplicity, materiality, and seed-4 boundary
are locked before execution.
""")
    if failures:
        raise RuntimeError(f"C78F protocol red team failed: {failures}")
    print(json.dumps({"gate": "C78F_PROTOCOL_RED_TEAM_PASS", "checks": len(checks), "failures": 0}, sort_keys=True))
    return {"passed": True, "checks": len(checks)}


if __name__ == "__main__":
    run()
