"""Finalize C77 only after independent red-team and regression verification."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from . import c77_independent_multiregime_replication_protocol as analysis
from . import c77_protocol


MAIN_MD = c77_protocol.REPORT_DIR / "C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.md"
RESULT_JSON = c77_protocol.REPORT_DIR / "C77_REPLICATION_PROTOCOL_RESULT.json"
THEORY_NOTE = c77_protocol.REPORT_DIR / "C77_THEORY_SCOPING_NOTE.md"
RED_TEAM = c77_protocol.REPORT_DIR / "C77_RED_TEAM_VERIFICATION.md"
HANDOFF = Path("oaci/OACI_CODEX_HANDOFF.md")
LOG_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs")


def _rows(name: str) -> list[dict]:
    with open(c77_protocol.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(c77_protocol.TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _latest_log(prefix: str) -> tuple[Path, Path, int]:
    paths = sorted(LOG_ROOT.glob(f"{prefix}_*.out"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError(f"C77 regression log missing: {prefix}")
    stdout = paths[-1]
    return stdout, stdout.with_suffix(".err"), int(stdout.stem.rsplit("_", 1)[1])


def _regressions() -> list[dict]:
    rows = []
    for suite, prefix in (
        ("focused_C77", "c77-reg-focused"),
        ("C65_C77", "c77-reg-c65"),
        ("C23_C77", "c77-reg-c23"),
        ("full_OACI", "c77-reg-full"),
    ):
        stdout, stderr, job_id = _latest_log(prefix)
        text = stdout.read_text()
        matches = re.findall(r"(\d+) passed(?:, [^\n]+)? in [0-9.]+s", text)
        if len(matches) != 1 or not stderr.is_file() or stderr.read_text() or "failed" in text.lower():
            raise RuntimeError(f"C77 regression did not pass cleanly: {suite} job {job_id}")
        rows.append({"suite": suite, "job_id": job_id, "passed_tests": int(matches[0]), "stderr_empty": 1, "status": "passed", "stdout_sha256": c77_protocol.sha256(stdout), "stderr_sha256": c77_protocol.sha256(stderr)})
    return rows


def _theory_note() -> str:
    return """# C77 Theory Scoping Note

## Object

Let `Y_{s,r,t,l,c}` be a held-out target utility for seed `s`, regime `r`, target `t`, trajectory/level `l`, and checkpoint candidate `c`. Let `X` be one of the prospectively registered strict-source, target-unlabeled, or split-label measurement blocks. C77 scopes, but does not prove, a heterogeneous model

```text
Y_{s,r,t,l,c} = a_{s,r,t,l} + f_{s,r,t,l}(X_c) + epsilon_{s,r,t,l,c}.
```

A nonzero within-group dependence between `X` and `Y` only establishes local measurement. Transport additionally requires stability of `f` across held-out targets, trajectories, regimes, and the seed-4 field. Control further requires enough separation at the extreme order statistic to improve top-k choice or regret.

## Extreme action

For candidate field size `M`, top-1 recovery depends on the best-versus-near-best gaps and effective near-tie multiplicity, not raw `M` alone. Even reliable bulk ordering can fail when many candidates occupy an epsilon-optimal set. C78/C79 must therefore report top gaps, effective multiplicity, random top-k baselines, and regret with association.

## Multi-regime transport

The synthetic benchmark varies local association, regime/target coefficient heterogeneity, candidate count, effective multiplicity, top-gap scale, and label budget. It is a design-calibration instrument. It demonstrates that the registered analysis can detect a stable signal and that heterogeneity can separate local association from transport. It is not evidence that real EEG follows this model.

## Identifiability boundary

Function-preserving latent reparameterizations remain valid. Orbit robustness cannot identify a W-versus-z causal origin. Strict-source, target-unlabeled, construction-label, evaluation-label, and same-label-oracle views remain separate information classes.

## Status

No theorem, EEG minimax bound, target-population claim, selector, or deployability result is asserted. Seed 3 is protocol-debug evidence; seed 4 is the future locked confirmation field. An external dataset remains a later, separately authorized stage.
"""


def _main_report(state: dict, regressions: list[dict]) -> str:
    power = {row["gate"]: row for row in _rows("power_and_false_positive_plan.csv")}
    storage = {row["campaign"]: row for row in _rows("compute_storage_plan.csv")}
    repairs = _rows("red_team_repair_ledger.csv")
    regression_text = ", ".join(f"{row['suite']} {row['passed_tests']} green (job {row['job_id']})" for row in regressions)
    return f"""# C77 — Independent Multi-Regime Instrumented Replication Protocol

**Final gate:** `{state['final_gate_candidate']}`

**Primary:** `{state['primary_candidate']}`

**Secondary active:** `C77-S1 + C77-S2 + C77-S3 + C77-S4 + C77-S5 + C77-S6 + C77-S7 + C77-S8`

## Gate-first result

- Protocol lock commit: `{state['protocol_commit'][:7]}`; C77 protocol SHA-256 `{state['protocol_sha256']}`.
- C78 seed-3 protocol SHA-256: `{state['C78_protocol_sha256']}`.
- Real training / EEG forward / re-inference / GPU / seed-3 / seed-4 / BNCI2014_004 access: `0 / 0 / 0 / 0 / 0 / 0 / 0`.
- Exact primary regime identities recovered: `ERM + OACI + SRC`.
- Comparable 40-checkpoint trajectories per level: `OACI + SRC`; ERM is a one-checkpoint shared stage-1 anchor.
- Registered levels: `0 + 1`; full field per seed: `1,458` checkpoint-target-level units.
- Seed-3 pilot: SHA-selected target `4`, regime `OACI`, both levels, shared ERM anchors, `82` units.

## Historical regime boundary

ERM and OACI are original pre-C14 regimes. SRC is not presented as an untouched method candidate: it was introduced after C10, fixed at `smooth_temperature=0.1` before C12, and C12 falsified its source-to-target transfer. C77 uses it only as a pre-existing, target-isolated negative-control trajectory. `global_lpc` and `uniform` are recoverable but excluded prospectively to avoid unnecessary regime multiplicity.

All seven regime/engine/manifest blobs checked by red team are byte-identical to their historical commits. Historical C11 evidence also replays `target_fit_ids_empty=true` and `selector target_read=false`.

## Synthetic power

The pre-committed 486-cell grid used 400 replicates per cell in 8 content-disjoint Slurm shards:

```text
null association FPR:                   {float(power['null_association_FPR']['observed']):.6f}
stable local-association power:          {float(power['stable_local_association_power']['observed']):.6f}
transport drop under heterogeneity:      {float(power['heterogeneity_reduces_transport']['observed']):.6f}
actionability drop at high multiplicity: {float(power['effective_multiplicity_reduces_actionability']['observed']):.6f}
```

The final contrast passes only the registered directional gate. Its `0.0075` magnitude is small and is not called material. Seed-3 must recalibrate this design effect before any seed-4 protocol is finalized.

## Compute and storage

```text
C78 pilot:        {storage['C78_seed3_P1']['retained_checkpoint_target_level_units']} units, {float(storage['C78_seed3_P1']['trial_cache_GiB_projected']):.3f} GiB cache, {storage['C78_seed3_P1']['GPU_hours_planning_low']}–{storage['C78_seed3_P1']['GPU_hours_planning_high']} GPU-hour budget range
seed-3 full:      {storage['C78_seed3_full']['retained_checkpoint_target_level_units']} units, {float(storage['C78_seed3_full']['trial_cache_GiB_projected']):.3f} GiB cache, {storage['C78_seed3_full']['GPU_hours_planning_low']}–{storage['C78_seed3_full']['GPU_hours_planning_high']} GPU-hour budget range
seed-3 + seed-4: {storage['R1_seed3_plus_seed4']['retained_checkpoint_target_level_units']} units, {float(storage['R1_seed3_plus_seed4']['trial_cache_GiB_projected']):.3f} GiB cache
```

The GPU numbers are conservative unmeasured planning ranges, not observed runtime. C78 P1 must measure and re-gate runtime before P2. Future GPU primary is the historically used `V100`; CPU instrumentation/analysis uses `cpu-high`, 48 cores. Availability is not authorization.

## Red team

Independent red team passed `60/60` blocking checks. It blocked one intermediate run because analysis mutated a protocol-hash-locked failure ledger; the locked bytes were restored and dynamic outcomes moved to `analysis_failure_reason_ledger.csv`. It also enforced:

""" + "\n".join(f"- `{row['item']}`: {row['resolution']}" for row in repairs) + f"""

The one nonblocking failure is the synthetic multiplicity materiality caveat above. Regression: {regression_text}.

## Decision

Independent instrumented training is now scientifically justified for checkpoint-field replication and cross-regime transport testing. It is **not authorized in C77**. No EEG hypothesis has replicated yet; no representation mechanism, target-population generalization, selector, checkpoint recommendation, or deployable control is claimed.

The pre-committed JSON at `C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.json` is intentionally not overwritten after compute. Post-compute gate evidence is in `C77_REPLICATION_PROTOCOL_RESULT.json` and the C77 tables.
"""


def _handoff_section(state: dict) -> str:
    return f"""## 0. Current continuation state (2026-07-10)

The detailed C23-C31 history below remains useful background, but the authoritative tip is now C77:

```text
C77 protocol commit: {state['protocol_commit'][:7]}
C77 protocol SHA:    {state['protocol_sha256']}
C77 final gate:      {state['final_gate_candidate']}
C77 primary:         {state['primary_candidate']}
```

C77 is protocol/power/readiness only. It recovered exact historical ERM/OACI/SRC identities, with OACI/SRC as the two comparable 40-checkpoint trajectories per level and ERM as a shared one-point stage-1 anchor. SRC remains the C12-falsified negative control, not a method-rescue candidate. Levels 0/1 yield 1,458 planned units per seed.

The 486-cell, 400-replicate synthetic benchmark passed its locked FPR/power/direction gates, but the effective-multiplicity actionability contrast is only 0.0075 and is not material. Independent red team passed 60/60 blocking checks after repairing a locked-ledger mutation. No training, real EEG forward, GPU work, seed-3/4 access, BNCI2014_004 access, checkpoint creation, selector, or manuscript drafting occurred.

C78 seed-3 is ready but not authorized. Only the future exact CLI token in the locked C78 protocol can authorize its 82-unit target-4/OACI pilot; prompt text and environment variables cannot. Seed 4 remains inaccessible until a final C79 protocol is committed, hashed, and separately authorized.

Authoritative artifacts:

```text
oaci/reports/C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.md
oaci/reports/C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.json
oaci/reports/C77_REPLICATION_PROTOCOL_RESULT.json
oaci/reports/C77_RED_TEAM_VERIFICATION.md
oaci/reports/C78_SEED3_INSTRUMENTED_PILOT_PROTOCOL.json
oaci/reports/c77_tables/artifact_manifest.csv
```

Do not start C78 training, use seed 3 or 4, access BNCI2014_004, create checkpoints, or draft manuscript text without a new explicit PM instruction and the exact future authorization interface.

---
"""


def _update_handoff(state: dict) -> None:
    text = HANDOFF.read_text()
    start = text.index("## 0. Current continuation state")
    end = text.index("\n## 1. What C23–C31 established", start)
    HANDOFF.write_text(text[:start] + _handoff_section(state) + "\n" + text[end:])


def _artifact_manifest() -> list[dict]:
    paths = [
        *sorted(c77_protocol.TABLE_DIR.rglob("*")),
        c77_protocol.PROTOCOL_PATH, c77_protocol.PROTOCOL_SHA_PATH,
        c77_protocol.TIMING_PATH, c77_protocol.C78_PROTOCOL_PATH,
        c77_protocol.C78_PROTOCOL_SHA_PATH, c77_protocol.C79_SKELETON_PATH,
        MAIN_MD, RESULT_JSON, THEORY_NOTE, RED_TEAM,
        analysis.STATE_PATH,
        Path("oaci/conditioned_ceiling_coverage/c77_protocol.py"),
        Path("oaci/conditioned_ceiling_coverage/c77_independent_multiregime_replication_protocol.py"),
        Path("oaci/conditioned_ceiling_coverage/synthetic_multiregime_generator.py"),
        Path("oaci/conditioned_ceiling_coverage/c77_red_team.py"),
        Path("oaci/conditioned_ceiling_coverage/c77_finalize.py"),
        Path("oaci/slurm_c77_protocol.sh"), Path("oaci/slurm_c77_synthetic.sh"),
        Path("oaci/slurm_c77_analyze.sh"), Path("oaci/slurm_c77_red_team.sh"),
        Path("oaci/slurm_c77_regression.sh"), Path("oaci/slurm_c77_finalize.sh"),
        Path("oaci/tests/test_c77_independent_multiregime_replication_protocol.py"),
    ]
    rows, seen = [], set()
    for path in paths:
        if path in seen or not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        seen.add(path)
        row_count = 0
        if path.suffix == ".csv":
            with open(path, newline="") as stream:
                row_count = sum(1 for _ in csv.DictReader(stream))
        rows.append({"path": str(path), "sha256": c77_protocol.sha256(path), "size_bytes": path.stat().st_size, "row_count": row_count, "raw_EEG_or_weights": 0, "tracked_or_to_be_tracked": 1})
    return rows


def finalize() -> dict:
    checks = _rows("red_team_checks.csv")
    if len(checks) != 61 or any(row["blocking"] == "1" and row["passed"] != "1" for row in checks):
        raise RuntimeError("C77 finalization requires 60/60 blocking red-team checks")
    if "Final status: `PASS`" not in RED_TEAM.read_text():
        raise RuntimeError("C77 red-team report is not PASS")
    regressions = _regressions()
    _write_csv("regression_verification.csv", regressions)
    state = json.loads(analysis.STATE_PATH.read_text())
    THEORY_NOTE.write_text(_theory_note())
    MAIN_MD.write_text(_main_report(state, regressions))
    result = {
        "schema_version": "c77_replication_protocol_result_v1",
        "milestone": "C77", "final_gate": state["final_gate_candidate"],
        "taxonomy": {"primary_active": [state["primary_candidate"]], "primary_inactive": ["C77-B_historical_regimes_not_recoverable", "C77-C_power_or_compute_insufficient", "C77-D_claim_or_target_isolation_blocker", "C77-E_independent_training_not_scientifically_justified"], "secondary_active": [key for key, value in state["secondary_candidates"].items() if value]},
        "protocol": {"commit": state["protocol_commit"], "sha256": state["protocol_sha256"], "C78_sha256": state["C78_protocol_sha256"], "prospective_before_synthetic": True},
        "execution_boundary": state["execution_boundary"],
        "regimes": {"primary": ["ERM", "OACI", "SRC"], "comparable_trajectories": ["OACI", "SRC"], "levels": [0, 1], "SRC_negative_control": True, "units_per_seed": 1458},
        "synthetic": {row["gate"]: {"observed": float(row["observed"]), "threshold": float(row["threshold"]), "passed": bool(int(row["passed"]))} for row in _rows("power_and_false_positive_plan.csv")},
        "red_team": {"blocking_passed": 60, "blocking_total": 60, "nonblocking_failed": 1, "repair_count": len(_rows("red_team_repair_ledger.csv"))},
        "regression": regressions,
        "claims": state["claims"],
    }
    RESULT_JSON.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    _update_handoff(state)
    manifest = _artifact_manifest()
    _write_csv("artifact_manifest.csv", manifest)
    large = [{"path": row["path"], "size_bytes": row["size_bytes"], "over_50MiB": int(int(row["size_bytes"]) > 50 * 1024 * 1024), "passed": int(int(row["size_bytes"]) <= 50 * 1024 * 1024)} for row in manifest]
    _write_csv("large_artifact_scan.csv", large)
    if any(row["over_50MiB"] for row in large):
        raise RuntimeError("C77 report artifact exceeds 50 MiB")
    print(json.dumps({"gate": result["final_gate"], "primary": result["taxonomy"]["primary_active"], "regressions": regressions, "artifacts": len(manifest)}, sort_keys=True))
    return result


if __name__ == "__main__":
    finalize()
