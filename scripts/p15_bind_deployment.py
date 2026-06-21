"""Bind the DEPLOYMENT-BRANCH redump (feat_dump_v4, full Freeze A1 grid) to Freeze A1 — ATOMIC, all-or-nothing.

Unlike p15_bind_redump (which binds a FIXED LPC config), this binds the ACTUAL deployed CITA+LPC arm: per cohort
the nested selector's chosen lambda, recovered from the frozen r12det_full CITA_nested pred-hashes (NOT hardcoded).
For seed 0 that is PD->lpc_prior:0.1, SCZ->lpc_prior:0.3. The no-LPC arm is erm:0 everywhere. Every shard's
predictions must align EXACTLY with Freeze A1 (the runner's float32 pred_hash), content hashes must verify, the
runner CODE must equal its committed version, and the two branches must share identical sample-ID set+order per
cohort. Partial success => NO manifest, nonzero exit; tolerances are never relaxed.
"""
import os, sys, json, glob, hashlib, subprocess
import numpy as np
# reuse the EXACT verified helpers from the fixed-config binder (same freeze source, same canonical hash)
from p15_bind_redump import expected_cohorts, freeze_pred_hash, canon_pred_hash16, FREEZE

V4 = "results/feat_dump_v4"
OUT = f"{V4}/FEATURE_BOUND_DEPLOYMENT.json"

def selected_config(cond, cohort):
    """Recover the nested-selected config for a cohort by matching the frozen CITA_nested pred_hash to the fixed
    configs' pred_hashes in r12det_full (seed 0). Returns 'erm:0' / 'lpc_prior:0.1' / 'lpc_prior:0.3' or None."""
    o = json.load(open(f"results/r12det_dualpc2/r12det_full_{cond}_s0.json"))
    F = o["folds"]; cfgs = [c for c in F if c != "CITA_nested"]
    nested = {r["held_out"]: r["pred_hash"] for r in F["CITA_nested"]}
    nh = nested.get(cohort)
    for c in cfgs:
        fixed = {r["held_out"]: r["pred_hash"] for r in F[c]}
        if fixed.get(cohort) == nh:
            return c
    return None

def _load_shard(cond, cohort, lbl):
    fn = f"{V4}/audit_{cond}_{cohort}_{lbl.replace(':', '_')}.npz"
    matches = glob.glob(fn)
    if len(matches) != 1:
        return None, f"{'MISSING' if not matches else 'DUPLICATE'} shard {cond}/{cohort}/{lbl}"
    return np.load(fn, allow_pickle=True), None

def _verify(o, cond, cohort, lbl, role):
    """All per-shard checks; returns (shard_dict, problem_or_None)."""
    fz_ph = freeze_pred_hash(cond, cohort, lbl); got_ph = canon_pred_hash16(o["prob_te"])
    if fz_ph != got_ph:
        return None, f"PRED-HASH MISMATCH {cond}/{cohort}/{lbl}: freeze={fz_ph} redump={got_ph}"
    dump_branch = "CITA-no-LPC" if lbl.startswith("erm") else "CITA+LPC"
    if str(o["branch"]) != dump_branch or str(o["selected_config_id"]) != lbl:
        return None, f"DUMP-LABEL MISMATCH {cond}/{cohort}/{lbl}: branch={o['branch']} id={o['selected_config_id']}"
    if not all("::" in str(g) for g in o["group_id_te"]):
        return None, f"GROUP-ID not namespaced {cond}/{cohort}/{lbl}"
    sids = [str(s) for s in o["sample_id_te"]]
    if len(sids) != len(set(sids)):
        return None, f"DUPLICATE sample_id {cond}/{cohort}/{lbl}"
    if str(o["freeze_a1_hash"]) != FREEZE["hash"] or str(o["scientific_code_commit"]) != FREEZE["code_commit"]:
        return None, f"PROVENANCE MISMATCH {cond}/{cohort}/{lbl}"
    return dict(cond=cond, cohort=cohort, config=lbl, role=role, selected_for_deployment=(role == "CITA+LPC-deployment"),
                instrumentation_commit=str(o["instrumentation_commit"]), head_commit=str(o.get("head_commit")),
                runner_file_sha=str(o.get("runner_file_sha")), env_sha=str(o.get("env_sha")), sample_ids_te=tuple(sids),
                feat_hash_te=str(o["feat_hash_te"]), feat_hash_se=str(o["feat_hash_se"]),
                feat_hash_ev=str(o["feat_hash_ev"]), pred_hash_te=got_ph, n_te=int(len(sids))), None

def main():
    if os.path.exists(OUT):
        print(f"{OUT} exists — refusing to overwrite (immutable)."); sys.exit(2)
    problems, shards, selection = [], [], {}
    for cond in ("PD", "SCZ"):
        cohorts = expected_cohorts(cond)
        if not cohorts:
            problems.append(f"{cond}: no Freeze A1 cohorts found"); continue
        for cohort in cohorts:
            sel = selected_config(cond, cohort)
            if sel is None:
                problems.append(f"UNRESOLVED selection {cond}/{cohort} (no fixed config matched CITA_nested)"); continue
            selection[f"{cond}/{cohort}"] = sel
            # the two deployed branches for this cohort: no-LPC (erm:0) and the ACTUAL selected arm
            want = [("erm:0", "CITA-no-LPC")]
            if sel == "erm:0":
                # deployment chose no LPC here -> the +LPC arm IS no-LPC; flag (degenerate contrast for this cohort)
                problems.append(f"NOTE {cond}/{cohort}: deployment selected erm:0 (no LPC) — excluded from +LPC contrast")
            else:
                want.append((sel, "CITA+LPC-deployment"))
            for lbl, role in want:
                o, err = _load_shard(cond, cohort, lbl)
                if err:
                    problems.append(err); continue
                shard, perr = _verify(o, cond, cohort, lbl, role)
                if perr:
                    problems.append(perr); continue
                shards.append(shard)
    # ---- code-immutability across ALL shards (single run) + runner content == committed ----
    for k in ("instrumentation_commit", "head_commit", "runner_file_sha", "env_sha"):
        if len(set(s[k] for s in shards)) > 1:
            problems.append(f"INCONSISTENT {k} across shards (tree changed mid-run): {sorted(set(s[k] for s in shards))}")
    if shards:
        head, rsha = shards[0]["head_commit"], shards[0]["runner_file_sha"]
        blob = subprocess.run(["git", "show", f"{head}:cmi/run_scps_crossdataset.py"], capture_output=True).stdout
        if (hashlib.sha256(blob).hexdigest() if blob else None) != rsha:
            problems.append(f"RUNNER CONTENT != committed @ {head[:12]} (uncommitted CODE change)")
    # ---- sample-ID set+ORDER identical across the two deployed branches per cohort (leakage diff != sample selection) ----
    by_fold = {}
    for s in shards:
        by_fold.setdefault((s["cond"], s["cohort"]), {})[s["role"]] = s
    n_contrast = 0
    for (cond, cohort), bs in by_fold.items():
        if "CITA+LPC-deployment" not in bs:                     # erm:0-selected cohort: no +LPC contrast (already noted)
            continue
        n_contrast += 1
        if bs["CITA-no-LPC"]["sample_ids_te"] != bs["CITA+LPC-deployment"]["sample_ids_te"]:
            problems.append(f"SAMPLE-ID set/ORDER DIFFERS across branches {cond}/{cohort}")
    hard = [p for p in problems if not p.startswith("NOTE")]
    if hard or n_contrast == 0:
        print(f"DEPLOYMENT REDUMP NOT BOUND — {len(shards)} shards, {n_contrast} +LPC contrasts; {len(hard)} problems:")
        for p in problems[:24]:
            print(f"  - {p}")
        print("NO FEATURE_BOUND_DEPLOYMENT manifest written. (Re-run failed jobs IDENTICALLY; do NOT relax.)")
        sys.exit(1)
    slim = []
    for s in shards:
        d = {k: v for k, v in s.items() if k != "sample_ids_te"}
        d["sample_ids_te_hash"] = hashlib.sha256(repr(s["sample_ids_te"]).encode()).hexdigest()[:16]
        slim.append(d)
    manifest = dict(status="FEATURE_BOUND", branch="CITA+LPC-deployment-mixture", freeze_a1_hash=FREEZE["hash"],
                    scientific_code_commit=FREEZE["code_commit"], instrumentation_commit=shards[0]["instrumentation_commit"],
                    head_commit=shards[0]["head_commit"], runner_file_sha=shards[0]["runner_file_sha"],
                    env_sha=shards[0]["env_sha"], n_shards=len(shards), n_lpc_contrasts=n_contrast,
                    deployment_selection_seed0=selection,
                    code_immutability="all shards share runner_file_sha+commit+env; runner==committed; verified",
                    sampleid_branch_identity="CITA-no-LPC sample-ID set+order == CITA+LPC-deployment per cohort; verified",
                    notes=[p for p in problems if p.startswith("NOTE")], shards=slim)
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    tmp = OUT + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)
    print(f"FEATURE_BOUND (deployment mixture): {len(shards)} shards, {n_contrast} +LPC contrasts, all aligned w/ Freeze A1.")
    print(f"  deployment selection (seed0): {selection}")
    print(f"  -> {OUT}")

if __name__ == "__main__":
    main()
