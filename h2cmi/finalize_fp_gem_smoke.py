"""Validate the one-unit P12 smoke without reading target performance."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from h2cmi import run_fp_gem as runner


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "h2cmi/results/fp_gem_main"
EXPECTED_SOURCE = "f21981a86a61ca0c5129c642a5ecaee301fff0a98466a3fa09d7f89c719b3c43"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stderr_gate(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    allowed = {
        "Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present",
        "Matplotlib is building the font cache; this may take a moment.",
    }
    unexpected = [line for line in lines if line not in allowed]
    return {
        "exists": path.exists(),
        "status": "empty" if path.exists() and not lines else (
            "known_harmless_warnings_only" if path.exists() and not unexpected else "real_or_unexpected_failure"
        ),
        "sha256": sha256_file(path) if path.exists() else "",
        "unexpected_lines": unexpected,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--submit-command", required=True)
    ap.add_argument("--raw-root", default="/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12")
    args = ap.parse_args()
    raw = Path(args.raw_root)
    smoke_path = raw / "smoke.json"
    stdout_path = raw / f"logs/smoke-{args.job_id}.out"
    stderr_path = raw / f"logs/smoke-{args.job_id}.err"
    payload = json.loads(smoke_path.read_text())
    proc = subprocess.run(
        ["squeue", "-h", "-j", args.job_id, "-o", "%i|%T|%P|%R"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"squeue smoke validation failed: {proc.stderr.strip()}")
    queue_absent = not proc.stdout.strip()
    stdout = stdout_path.read_text() if stdout_path.exists() else ""
    stderr = stderr_gate(stderr_path)
    checks = {
        "job_absent_from_squeue": queue_absent,
        "smoke_payload_pass": payload.get("status") == "pass" and payload.get("mode") == "smoke",
        "exact_unit": (payload.get("dataset"), payload.get("target_subject"), payload.get("source_seed")) == (
            "BNCI2014_001", 1, 0
        ),
        "p9_reference_state_matches_committed_row": payload["source_checkpoint"]["p9_reference_source_model_sha256"] == EXPECTED_SOURCE,
        "source_reproduction_mode": payload["source_checkpoint"]["source_reproduction_mode"] == "exact_p9_configuration_retrain",
        "p9_checkpoint_file_unavailable": not payload["source_checkpoint"]["p9_checkpoint_file_available"],
        "actual_source_state_hash_complete": len(payload["source_checkpoint"]["source_model_sha256_actual"]) == 64,
        "checkpoint_file_checksum": sha256_file(payload["source_checkpoint"]["source_checkpoint_path"])
        == payload["source_checkpoint"]["source_checkpoint_file_sha256"],
        "feature_dimension": payload["feature_hook"]["dimension"] == 210,
        "feature_hook_semantics": max(
            payload["feature_hook"]["source_semantic_max_abs_error"],
            payload["feature_hook"]["adapt_semantic_max_abs_error"],
            payload["feature_hook"]["eval_semantic_max_abs_error"],
        ) <= 1e-7,
        "six_method_prediction_shapes": set(payload["method_prediction_shapes"]) == set(runner.P9_METHODS) | set(runner.NEW_METHODS)
        and all(shape == [72, 2] for shape in payload["method_prediction_shapes"].values()),
        "six_method_prediction_hashes_complete": set(payload["method_prediction_hashes"]) == set(runner.P9_METHODS) | set(runner.NEW_METHODS)
        and all(payload["method_prediction_hashes"].values()),
        "six_method_logits_hashes_complete": set(payload["method_logits_hashes"]) == set(runner.P9_METHODS) | set(runner.NEW_METHODS)
        and all(payload["method_logits_hashes"].values()),
        "six_methods_share_source_state": payload["control_reproduction"]["reproduced_source_model_sha256"]
        == payload["source_checkpoint"]["source_model_sha256_actual"],
        "p9_rows_not_reused": not payload["control_reproduction"]["p9_rows_reused"],
        "no_performance_metrics": not payload["performance_metrics_computed"],
        "evaluation_labels_not_accessed": not payload["evaluation_labels_accessed"],
        "target_labels_not_passed": not payload["target_labels_passed_to_adaptation"],
        "no_target_selection": not payload["target_performance_selection"],
        "classifier_frozen": payload["rct"]["classifier_sha256_before_rct"]
        == payload["rct"]["classifier_sha256_after_rct"],
        "parameters_frozen": payload["rct"]["parameters_sha256_before_rct"]
        == payload["rct"]["parameters_sha256_after_rct"],
        "fp_prior_fixed": payload["geometry"]["fp_pi_fit"] == payload["geometry"]["source_empirical_prior"],
        "clean_launch": payload["provenance"]["clean_worktree"] and payload["provenance"]["git_status_porcelain"] == "",
        "runner_hash": payload["provenance"]["runner_sha256"] == sha256_file(runner.__file__),
        "config_hash": payload["provenance"]["config_sha256"] == runner.FROZEN_CONFIG_SHA256,
        "stdout_exists": stdout_path.exists() and bool(stdout),
        "stdout_clean_header": "repo_status_porcelain_begin\nrepo_status_porcelain_end" in stdout,
        "stderr_accepted": stderr["status"] in {"empty", "known_harmless_warnings_only"},
    }
    smoke_pass = all(checks.values())
    report = {
        "status": "pass" if smoke_pass else "blocked",
        "approve_p12b_fleet": smoke_pass,
        "job_id": args.job_id,
        "submit_command": args.submit_command,
        "final_squeue": {
            "command": f"squeue -h -j {args.job_id} -o %i|%T|%P|%R",
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "absent": queue_absent,
        },
        "checks": checks,
        "smoke_payload_path": str(smoke_path),
        "smoke_payload_sha256": sha256_file(smoke_path),
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path) if stdout_path.exists() else "",
        "stderr": stderr | {"path": str(stderr_path)},
        "source_checkpoint": payload["source_checkpoint"],
        "feature_hook": payload["feature_hook"],
        "runtime": payload["provenance"]["runtime"],
        "performance_metrics_recorded": [],
    }
    (OUT / "fp_gem_smoke_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# FP-GEM Smoke Audit",
        "",
        f"- status: `{'PASS' if smoke_pass else 'BLOCKED'}`",
        f"- approve P12B fleet: `{smoke_pass}`",
        f"- Slurm job: `{args.job_id}`",
        f"- final squeue absence: `{queue_absent}`",
        f"- source state SHA-256: `{payload['source_checkpoint']['source_model_sha256_actual']}`",
        f"- source checkpoint file SHA-256: `{payload['source_checkpoint']['source_checkpoint_file_sha256']}`",
        f"- feature dimension: `{payload['feature_hook']['dimension']}`",
        f"- maximum hook replay error: `{max(payload['feature_hook']['source_semantic_max_abs_error'], payload['feature_hook']['adapt_semantic_max_abs_error'], payload['feature_hook']['eval_semantic_max_abs_error'])}`",
        f"- evaluation labels accessed: `{payload['evaluation_labels_accessed']}`",
        f"- performance metrics computed: `{payload['performance_metrics_computed']}`",
        f"- stderr status: `{stderr['status']}`",
        "",
        "The smoke validated only exact-config source retraining, persisted checkpoint provenance, six-method checkpoint identity, feature-hook semantics, shapes, finite transforms/logits, frozen parameters, and leakage boundaries. Accuracy and balanced accuracy were neither computed nor recorded and cannot influence the frozen P12 configuration.",
        "",
        "## Gate Checks",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    (OUT / "fp_gem_smoke_audit.md").write_text("\n".join(lines) + "\n")
    integration = OUT / "fp_gem_integration_audit.md"
    text = integration.read_text().replace("- GPU smoke status: `PENDING`", f"- GPU smoke status: `{'PASS' if smoke_pass else 'BLOCKED'}`")
    text = text.replace("- approve P12B fleet before smoke: `false`", f"- approve P12B fleet after smoke: `{str(smoke_pass).lower()}`")
    text += (
        "\n## GPU Smoke Gate\n\n"
        f"- job id: `{args.job_id}`\n"
        f"- status: `{'PASS' if smoke_pass else 'BLOCKED'}`\n"
        f"- smoke payload SHA-256: `{sha256_file(smoke_path)}`\n"
        f"- exact P9 source-training configuration reproduced: `{checks['source_reproduction_mode']}`\n"
        f"- P9 reference state hash matched by retrain (informational, not a gate): `{payload['source_checkpoint']['p9_state_hash_matches_actual']}`\n"
        f"- all six methods share the reproduced state: `{checks['six_methods_share_source_state']}`\n"
        f"- feature-hook replay passed: `{checks['feature_hook_semantics']}`\n"
        f"- target performance observed: `false`\n"
    )
    integration.write_text(text)
    command_log = OUT / "COMMAND_LOG.md"
    command_log.write_text(
        command_log.read_text().rstrip()
        + "\n\n"
        + f"- P12A smoke submission: `{args.submit_command}`. Job `{args.job_id}` left `squeue`; "
        + f"artifact/hash/hook/leakage gates returned `{'PASS' if smoke_pass else 'BLOCKED'}`. "
        + "No target performance metric was computed.\n"
    )
    print(json.dumps({"status": report["status"], "approve_p12b_fleet": smoke_pass}, sort_keys=True))
    return 0 if smoke_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
