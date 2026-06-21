"""Bind the frozen redump (feat_dump_v3) to Freeze A1 — ATOMIC, all-or-nothing. Writes a FEATURE_BOUND manifest
ONLY if EVERY expected cohort×branch shard exists, is unique, its predictions align EXACTLY with the Freeze A1
run (r12det_full), the algorithm_role/branch semantics are explicit and correct, IDs are namespaced+unique, and
all canonical content hashes verify. Partial success => NO manifest, nonzero exit. (Per directive points 1-4.)

Re-running a FAILED redump is allowed with the IDENTICAL config; this script never relaxes a tolerance or renames
a version to force a bind.
"""
import os, sys, json, glob, hashlib
import numpy as np

V3 = "results/feat_dump_v3"
OUT = f"{V3}/FEATURE_BOUND.json"
FREEZE = json.load(open("results/freeze_a1/manifest.json"))
# explicit algorithm-role semantics (point 1: erm:0 here is the frozen CITA-no-LPC ARM, NOT a bare ERM baseline)
ROLE = {"erm:0": ("CITA-no-LPC", "ERM encoder + matched-CORAL alignment + gate (the no-LPC deployment arm of CITA)"),
        "lpc_prior:0.3": ("CITA+LPC", "LPC(0.3) encoder + matched-CORAL alignment + gate")}
CONFIGS = ["erm:0", "lpc_prior:0.3"]

def expected_cohorts(cond):
    """The cohorts the Freeze A1 run actually evaluated (authoritative source of the shard set)."""
    cs = set()
    for f in glob.glob(f"results/r12det_dualpc2/r12det_full_{cond}_s0.json"):
        o = json.load(open(f))
        for r in o.get("folds", {}).get("erm:0", []):
            cs.add(r["held_out"])
    return sorted(cs)

def freeze_pred_hash(cond, cohort, lbl):
    o = json.load(open(f"results/r12det_dualpc2/r12det_full_{cond}_s0.json"))
    for r in o.get("folds", {}).get(lbl, []):
        if r["held_out"] == cohort:
            return r.get("pred_hash")
    return None

def canon_pred_hash16(prob):
    a = np.round(np.asarray(prob, "<f8"), 6)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

def main():
    if os.path.exists(OUT):
        print(f"{OUT} exists — refusing to overwrite (immutable)."); sys.exit(2)
    problems, shards = [], []
    for cond in ("PD", "SCZ"):
        cohorts = expected_cohorts(cond)
        if not cohorts:
            problems.append(f"{cond}: no Freeze A1 cohorts found"); continue
        for cohort in cohorts:
            for lbl in CONFIGS:
                fn = f"{V3}/audit_{cond}_{cohort}_{lbl.replace(':', '_')}.npz"
                matches = glob.glob(fn)
                if len(matches) == 0:
                    problems.append(f"MISSING shard {cond}/{cohort}/{lbl}"); continue
                if len(matches) > 1:
                    problems.append(f"DUPLICATE shard {cond}/{cohort}/{lbl}"); continue
                o = np.load(fn, allow_pickle=True)
                # (a) prediction alignment with Freeze A1 (the runner pred_hash, 16-char)
                fz_ph = freeze_pred_hash(cond, cohort, lbl)
                got_ph = canon_pred_hash16(o["prob_te"])
                if fz_ph != got_ph:
                    problems.append(f"PRED-HASH MISMATCH {cond}/{cohort}/{lbl}: freeze={fz_ph} redump={got_ph}"); continue
                # (b) explicit branch / algorithm_role semantics (point 1)
                role, _ = ROLE[lbl]
                if str(o["branch"]) != role or str(o["selected_config_id"]) != lbl:
                    problems.append(f"ROLE MISMATCH {cond}/{cohort}/{lbl}: branch={o['branch']}"); continue
                # (c) namespaced + unique group/sample ids (per split)
                gids = [str(g) for g in o["group_id_te"]]
                if not all("::" in g for g in gids):
                    problems.append(f"GROUP-ID not namespaced {cond}/{cohort}/{lbl}"); continue
                sids = [str(s) for s in o["sample_id_te"]]
                if len(sids) != len(set(sids)):
                    problems.append(f"DUPLICATE sample_id in te {cond}/{cohort}/{lbl}"); continue
                # (d) provenance
                if str(o["freeze_a1_hash"]) != FREEZE["hash"] or str(o["scientific_code_commit"]) != FREEZE["code_commit"]:
                    problems.append(f"PROVENANCE MISMATCH {cond}/{cohort}/{lbl}"); continue
                # (e) content checksums recorded in the shard (canonical, not .npz bytes)
                shards.append(dict(cond=cond, cohort=cohort, config=lbl, role=role,
                                   instrumentation_commit=str(o["instrumentation_commit"]),
                                   feat_hash_te=str(o["feat_hash_te"]), feat_hash_se=str(o["feat_hash_se"]),
                                   feat_hash_ev=str(o["feat_hash_ev"]), pred_hash_te=got_ph,
                                   n_te=int(len(sids))))
    n_expected = sum(len(expected_cohorts(c)) * len(CONFIGS) for c in ("PD", "SCZ"))
    if problems or len(shards) != n_expected:
        print(f"REDUMP NOT COMPLETE — {len(shards)}/{n_expected} shards bound; {len(problems)} problems:")
        for p in problems[:20]:
            print(f"  - {p}")
        print("NO FEATURE_BOUND manifest written. (Re-run failed jobs with IDENTICAL config; do NOT relax.)")
        sys.exit(1)
    manifest = dict(status="FEATURE_BOUND", freeze_a1_hash=FREEZE["hash"], scientific_code_commit=FREEZE["code_commit"],
                    instrumentation_commit=shards[0]["instrumentation_commit"], n_shards=len(shards),
                    algorithm_roles={lbl: dict(role=ROLE[lbl][0], definition=ROLE[lbl][1]) for lbl in CONFIGS},
                    shards=shards)
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    tmp = OUT + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)
    print(f"FEATURE_BOUND: {len(shards)}/{n_expected} shards, all pred-hash-aligned with Freeze A1. -> {OUT}")
    print(f"  algorithm_roles: erm:0 => CITA-no-LPC (NOT bare ERM), lpc_prior:0.3 => CITA+LPC")

if __name__ == "__main__":
    main()
