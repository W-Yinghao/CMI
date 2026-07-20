"""Independent metadata-only red team for the prospective C78R protocol."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import c78r_common as common
from . import c78r_seed3_src_canary as c78r


REPORT = c78r.REPORT_DIR / "C78R_PROTOCOL_RED_TEAM_VERIFICATION.md"
CHECKS = c78r.TABLE_DIR / "protocol_red_team_checks.csv"


def _check(rows, name, passed, observed, expected):
    rows.append({"check": name, "passed": int(bool(passed)), "blocking": 1, "observed": observed, "expected": expected})


def run() -> list[dict]:
    protocol, digest, token = common.load_protocol()
    units = c78r.read_csv(c78r.TABLE_DIR / "SRC_unit_manifest.csv")
    train_source = Path("oaci/conditioned_ceiling_coverage/c78r_train.py").read_text()
    train_function = train_source[train_source.index("def train_src_canary"):]
    rows = []
    _check(rows, "protocol_sha_replay", digest == c78r.PROTOCOL_SHA_PATH.read_text().strip(), digest, "committed SHA file")
    _check(rows, "parent_result_exact", protocol["parent_result_commit"] == c78r.PARENT_RESULT_COMMIT, protocol["parent_result_commit"], c78r.PARENT_RESULT_COMMIT)
    _check(rows, "exact_token_field", protocol["authorization_token_exact"] == c78r.AUTHORIZATION_TOKEN, token, c78r.AUTHORIZATION_TOKEN)
    _check(rows, "exact_CLI_only", protocol["authorization"]["accepted_channel"] == "exact_CLI_argument_only", protocol["authorization"], "CLI exact")
    _check(rows, "prompt_not_authorization", protocol["authorization"]["prompt_text_is_authorization"] is False, protocol["authorization"]["prompt_text_is_authorization"], False)
    _check(rows, "environment_not_authorization", protocol["authorization"]["environment_is_authorization"] is False, protocol["authorization"]["environment_is_authorization"], False)
    _check(rows, "unit_count", len(units) == 80, len(units), 80)
    _check(rows, "unit_unique", len({row["unit_id"] for row in units}) == 80, len({row["unit_id"] for row in units}), 80)
    _check(rows, "SRC_only", {row["regime"] for row in units} == {"SRC"}, {row["regime"] for row in units}, {"SRC"})
    _check(rows, "target_exact", {row["target"] for row in units} == {"4"}, {row["target"] for row in units}, {"4"})
    _check(rows, "seed_exact", {row["seed"] for row in units} == {"3"}, {row["seed"] for row in units}, {"3"})
    _check(rows, "levels_exact", {row["level"] for row in units} == {"0", "1"}, {row["level"] for row in units}, {"0", "1"})
    _check(rows, "temperature_exact", {row["smooth_temperature"] for row in units} == {"0.1"}, {row["smooth_temperature"] for row in units}, {"0.1"})
    _check(rows, "fixed_cadence", all([int(row["epoch"]) for row in units if row["level"] == str(level)] == list(c78r.SRC_EPOCHS) for level in c78r.LEVELS), "two 40-point sequences", list(c78r.SRC_EPOCHS))
    _check(rows, "historical_byte_replay", all(int(row["byte_exact"]) == 1 for row in protocol["historical_hashes"]), [row["byte_exact"] for row in protocol["historical_hashes"]], "all 1")
    _check(rows, "two_read_only_ERM_anchors", len(protocol["frozen_erm_initialization"]["anchors"]) == 2, len(protocol["frozen_erm_initialization"]["anchors"]), 2)
    _check(rows, "ERM_not_retrained", protocol["execution_boundary"]["ERM_retraining"] is False and "run_stage1_once" not in train_function, protocol["execution_boundary"]["ERM_retraining"], False)
    _check(rows, "OACI_not_retrained", protocol["execution_boundary"]["OACI_retraining"] is False and 'make_objective("OACI"' not in train_function, protocol["execution_boundary"]["OACI_retraining"], False)
    _check(rows, "guard_before_torch", train_function.index("common.require_authorization") < train_function.index("import torch"), "authorization before torch", "authorization before torch")
    _check(rows, "guard_before_EEG_loader", train_function.index("common.require_authorization") < train_function.index("from oaci.data.eeg.bnci"), "authorization before loader", "authorization before loader")
    _check(rows, "SRC_objective_only", "SRCObjective" in train_function and "train_stage2" in train_function, "SRCObjective/train_stage2", "SRCObjective/train_stage2")
    _check(rows, "full_domain_alignment", "plans.full_domain_alignment" in train_function, "present", "present")
    _check(rows, "C78_root_distinct", str(c78r.EXTERNAL_ROOT) != "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot", c78r.EXTERNAL_ROOT, "distinct")
    _check(rows, "C78_overwrite_forbidden", protocol["execution_boundary"].get("C78_artifact_overwrite", False) is False, protocol["execution_boundary"].get("C78_artifact_overwrite", False), False)
    _check(rows, "seed4_forbidden", protocol["execution_boundary"]["seed4"] is False, protocol["execution_boundary"]["seed4"], False)
    _check(rows, "BNCI004_forbidden", protocol["execution_boundary"]["BNCI2014_004"] is False, protocol["execution_boundary"]["BNCI2014_004"], False)
    _check(rows, "full_expansion_forbidden", protocol["execution_boundary"]["full_seed3_expansion"] is False, protocol["execution_boundary"]["full_seed3_expansion"], False)
    _check(rows, "remaining_scope_exact", protocol["expansion_gate"]["remaining_units"] == 1296 and protocol["expansion_gate"]["remaining_training_phases"] == 48, protocol["expansion_gate"], "1296/48")
    _check(rows, "implementation_hashes", all(c78r.sha256_file(item["path"]) == item["sha256"] for item in protocol["implementation_files"]), len(protocol["implementation_files"]), len(c78r.IMPLEMENTATION_FILES))
    _check(rows, "training_script_V100", "#SBATCH --partition=V100" in Path("oaci/slurm_c78r_train.sh").read_text(), "V100", "V100")
    _check(rows, "instrumentation_cpu_high", "#SBATCH --partition=cpu-high" in Path("oaci/slurm_c78r_instrument.sh").read_text(), "cpu-high", "cpu-high")
    _check(rows, "no_raw_payload_in_git", protocol["execution_boundary"]["raw_cache_or_weights_in_git"] is False, False, False)
    _check(rows, "no_selector", protocol["execution_boundary"]["selector_or_checkpoint_recommendation"] is False, False, False)
    _check(rows, "no_manuscript", protocol["execution_boundary"]["manuscript"] is False, False, False)
    c78r.write_csv(CHECKS, rows)
    failures = [row for row in rows if row["blocking"] and not row["passed"]]
    REPORT.write_text(
        "# C78R Protocol Red-Team Verification\n\n"
        f"Final status: `{'PASS' if not failures else 'FAIL'}`\n\n"
        f"- Blocking checks: `{len(rows) - len(failures)}/{len(rows)}`.\n"
        "- EEG data rows accessed: `0`.\n"
        "- GPU jobs submitted: `0`.\n"
        "- Target outcomes read: `0`.\n\n"
        "The only pre-freeze C78 dependency is the pair of hash-locked ERM weights required to initialize historical SRC stage-2. ERM/OACI are not retrained or overwritten; OACI weights are inaccessible to the worker.\n"
    )
    if failures:
        raise RuntimeError(f"C78R protocol red-team failed: {[row['check'] for row in failures]}")
    print(json.dumps({"status": "PASS", "blocking": len(rows)}, sort_keys=True))
    return rows


if __name__ == "__main__":
    run()
