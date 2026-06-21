"""P1.5 audit — deterministic, freeze-gated: does LPC's measured domain-leakage reduction reflect GENUINE
suppression of extractable domain information, or representation COLLAPSE?

NORMATIVE source: notes/FREEZE_PROTOCOL.md §4 (P1.5 retain/drop + saturation). This script never invents or
mutates thresholds, never edits CITA/LPC/ERM/alignment/selection/feature-gen code, never uses TUAB, never uses
target labels in any deployment-facing path. Offline TASK labels are used ONLY in the representation-integrity /
utility audit (clearly separated below from any deployment/source-free path — this audit has NO deployment path).

EXECUTION GATE: real-data analysis of results/feat_dump_v2/ is allowed ONLY if a valid Freeze A1 marker+manifest
exist. If absent: implement+test, schema-only validate the dumps, fit NO probes, write NO decision file, exit with
status BLOCKED_ON_FREEZE_A1 (nonzero is NOT used for the block — block is a clean, expected terminal state).

Decision is exactly one of: RETAIN_LPC | DROP_LPC_COLLAPSE | INCONCLUSIVE.
"""
import os, sys, json, glob, hashlib, tempfile, shutil
import numpy as np

PROTOCOL = "notes/FREEZE_PROTOCOL.md"
DUMP_DIR = "results/feat_dump_v2"
FREEZE_A1_CANDIDATES = ["results/freeze_a1/manifest.json", "FREEZE_A1.json", "notes/FREEZE_A1.json"]

# ---- pre-registered thresholds (READ-ONLY mirror of FREEZE_PROTOCOL.md §4; verified against it at runtime) ----
THRESH = dict(
    util_loss_pp_max=2.0,        # §4.3 task-utility loss <= 2.0 pp held-out task-probe bAcc
    eff_rank_drop_frac_max=0.15, # §4.4 effective-rank drop <= 15%
    scatter_drop_frac_max=0.15,  # §4.4 between/within scatter drop <= 15%
    feat_var_floor=1e-4,         # §4.4 feature-var not ~0
    sat_delta_kl=0.02,           # §4 saturation: 2 consecutive tiers improve attacker leakage by < 0.02 KL
    alt_probe_recover_frac=0.5,  # §4.2 alternative-bias probe must NOT recover > 50% of the leakage gap
)

def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def verify_protocol_thresholds():
    """Fail loudly if the mirrored thresholds are not literally present in the normative protocol file."""
    if not os.path.exists(PROTOCOL):
        return False, f"missing {PROTOCOL}"
    txt = open(PROTOCOL).read()
    need = ["2.0 pp", "15%", "δ=0.02", "recover most"]
    missing = [n for n in need if n not in txt]
    return (not missing), ("ok" if not missing else f"protocol missing tokens {missing}")

# ============================ A. FREEZE + INPUT-INTEGRITY GATE ============================
def freeze_gate():
    for p in FREEZE_A1_CANDIDATES:
        if os.path.exists(p):
            try:
                m = json.load(open(p))
            except Exception as e:
                return False, None, f"Freeze A1 marker {p} unreadable: {e}"
            if m.get("status") == "FROZEN" and m.get("freeze") in ("A1", "Freeze A1") and m.get("hash"):
                return True, m, p
            return False, None, f"Freeze A1 marker {p} present but invalid (status/freeze/hash)"
    return False, None, "no Freeze A1 marker"

def schema_validate(dump_dir):
    """Schema-ONLY (no probe fitting): what the dumps support, and what metadata is MISSING for the full audit."""
    files = sorted(glob.glob(f"{dump_dir}/*.npz"))
    report = dict(n_files=len(files), files=[os.path.basename(f) for f in files], pairs={}, present_keys=None,
                  missing_for_full_audit=[], integrity=dict())
    if not files:
        report["missing_for_full_audit"].append("no dumps")
        return report
    # parse cohort/config from filename; pair ERM <-> LPC per cohort
    by = {}
    for f in files:
        b = os.path.basename(f)[len("feat_"):-len(".npz")]
        cond = b.split("_")[0]; rest = b[len(cond) + 1:]
        if "_erm_0" in rest:
            coh, cfg = rest.replace("_erm_0", ""), "erm"
        elif "_lpc_prior_0.3" in rest:
            coh, cfg = rest.replace("_lpc_prior_0.3", ""), "lpc"
        else:
            continue
        by.setdefault((cond, coh), {})[cfg] = f
    keys0 = None
    for (cond, coh), d in sorted(by.items()):
        ok = "erm" in d and "lpc" in d
        report["pairs"][f"{cond}/{coh}"] = sorted(d)
        if ok:
            oe, ol = np.load(d["erm"]), np.load(d["lpc"])
            keys0 = keys0 or sorted(oe.keys())
            # dimensional + pairing checks (schema-level; no probe fitting)
            same_y = ("y_te" in oe and "y_te" in ol and oe["y_te"].shape == ol["y_te"].shape)
            report["integrity"][f"{cond}/{coh}"] = dict(
                erm_shapes={k: list(oe[k].shape) for k in oe.keys()},
                y_te_matched=bool(same_y and np.array_equal(oe["y_te"], ol["y_te"])))
    report["present_keys"] = keys0
    # metadata gaps for the FULL domain-leakage audit
    gaps = []
    if keys0 and "d_se" not in keys0 and "dom_se" not in keys0:
        gaps.append("DOMAIN/COHORT labels absent inside npz (z_se domain ids) — domain-probe needs them; "
                    "cohort is derivable from filename for TARGET features only")
    if keys0 and not any(k in keys0 for k in ("subj_se", "subject", "rec_id")):
        gaps.append("SUBJECT/RECORDING ids absent — GROUPED split (§ grouped) cannot be formed from the dumps")
    if keys0 and "sample_id" not in keys0:
        gaps.append("SAMPLE IDs absent — exact cross-run sample pairing relies on canonical row order only")
    if keys0 and not any(k in keys0 for k in ("prob_te", "logits_te")):
        gaps.append("PREDICTION vectors/logits absent — cannot hash-compare predictions inside dumps")
    report["missing_for_full_audit"] = gaps
    return report

# ============================ B. DOMAIN-PROBE AUDIT (real-data; gated) ============================
def _det_probe(name):
    """Deterministic domain probes of increasing capacity / differing inductive bias."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    if name == "linear":
        return LogisticRegression(max_iter=2000, C=1.0)
    if name == "mlp64":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0)
    if name == "mlp256x2":
        return MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=300, random_state=0)
    if name == "gbt":
        return HistGradientBoostingClassifier(random_state=0)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15)
    raise ValueError(name)

PROBE_TIERS = ["linear", "mlp64", "mlp256x2"]   # capacity ladder (saturation checked on these)
PROBE_ALT = ["gbt", "knn"]                       # alternative inductive biases

def domain_probe_audit(Z, D, Y, groups, tiers, rng):
    """Fit domain probes predicting D from Z (within Y is handled by the caller's grouped/conditional design).
    GROUPED split by `groups` (subject/recording); preprocessing fit on train only; val selects nothing here
    (fixed grid); reports train/val/heldout. NEVER uses target task labels for D (D = domain membership)."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    tr, te = next(gss.split(Z, D, groups))
    sc = StandardScaler().fit(Z[tr]); Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
    out = {}
    for t in tiers:
        clf = _det_probe(t).fit(Ztr, D[tr])
        p_te = clf.predict_proba(Zte)
        out[t] = dict(
            heldout_auroc=float(roc_auc_score(D[te], p_te[:, 1]) if len(set(D)) == 2 else np.nan),
            heldout_bacc=float(balanced_accuracy_score(D[te], clf.predict(Zte))),
            heldout_ce=float(log_loss(D[te], p_te, labels=sorted(set(D)))),
            train_bacc=float(balanced_accuracy_score(D[tr], clf.predict(Ztr))))
        out[t]["train_val_gap"] = out[t]["train_bacc"] - out[t]["heldout_bacc"]
    return out

# ============================ C. REPRESENTATION-INTEGRITY + UTILITY (real-data; gated) ============================
def representation_metrics(Z, Y, rng):
    """OFFLINE task labels used here ONLY (integrity/utility audit; NOT a deployment path)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    Z = np.asarray(Z, float); Zc = Z - Z.mean(0)
    s = np.linalg.svd(Zc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    stable_rank = float((s ** 2).sum() / (s[0] ** 2 + 1e-12))
    mu = Z.mean(0); Sb = Sw = 0.0
    for c in np.unique(Y):
        Zc2 = Z[Y == c]; mc = Zc2.mean(0)
        Sb += len(Zc2) * float(((mc - mu) ** 2).sum()); Sw += float(((Zc2 - mc) ** 2).sum())
    idx = rng.permutation(len(Z)); cut = int(0.7 * len(Z))
    clf = LogisticRegression(max_iter=1000).fit(Z[idx[:cut]], Y[idx[:cut]])
    task_bacc = float(balanced_accuracy_score(Y[idx[cut:]], clf.predict(Z[idx[cut:]])) * 100)
    return dict(eff_rank=eff_rank, stable_rank=stable_rank, scatter_ratio=Sb / (Sw + 1e-12),
                feat_norm=float(np.linalg.norm(Z, axis=1).mean()), feat_var=float(Z.var(0).mean()),
                centroid_sep=float(np.linalg.norm(np.diff([Z[Y == c].mean(0) for c in np.unique(Y)], axis=0))),
                task_bacc=task_bacc)

# ============================ E. PRE-REGISTERED DECISION ENGINE ============================
def decision_engine(leak_erm, leak_lpc, alt_erm, alt_lpc, rep_erm, rep_lpc, saturated):
    """Returns (decision, predicates). leak_* = strongest-tier held-out domain bAcc (lower=less leakage).
    rep_* = representation_metrics dicts. saturated = bool from the saturation rule. All thresholds from THRESH."""
    P = []
    def pred(name, value, thr, ok, prov):
        P.append(dict(predicate=name, value=round(float(value), 4) if value is not None else None,
                      threshold=thr, pass_=bool(ok), provenance=prov))
        return ok
    # (1) leakage reduction survives the strongest validation-selected probe
    leak_gap = leak_erm - leak_lpc
    c1 = pred("leakage_reduction_strong_probe", leak_gap, ">0", leak_gap > 0, "FREEZE_PROTOCOL §4.1")
    # (2) alternative-bias probe does NOT recover most of the gap
    alt_gap = alt_erm - alt_lpc
    recover = (alt_gap / leak_gap) if leak_gap > 1e-9 else 1.0
    c2 = pred("alt_probe_not_recovering", recover, f"<= leaves >{1-THRESH['alt_probe_recover_frac']:.0%} suppressed",
              recover >= THRESH["alt_probe_recover_frac"], "FREEZE_PROTOCOL §4.2")
    # (3) task-utility loss within bound
    util_loss = rep_erm["task_bacc"] - rep_lpc["task_bacc"]
    c3 = pred("task_utility_loss_pp", util_loss, f"<= {THRESH['util_loss_pp_max']}",
              util_loss <= THRESH["util_loss_pp_max"], "FREEZE_PROTOCOL §4.3")
    # (4) NOT collapse: eff-rank, scatter, feat-var
    er_drop = 1 - rep_lpc["eff_rank"] / (rep_erm["eff_rank"] + 1e-12)
    sc_drop = 1 - rep_lpc["scatter_ratio"] / (rep_erm["scatter_ratio"] + 1e-12)
    c4a = pred("eff_rank_drop_frac", er_drop, f"<= {THRESH['eff_rank_drop_frac_max']}",
               er_drop <= THRESH["eff_rank_drop_frac_max"], "FREEZE_PROTOCOL §4.4")
    c4b = pred("scatter_drop_frac", sc_drop, f"<= {THRESH['scatter_drop_frac_max']}",
               sc_drop <= THRESH["scatter_drop_frac_max"], "FREEZE_PROTOCOL §4.4")
    c4c = pred("feat_var_not_zero", rep_lpc["feat_var"], f">= {THRESH['feat_var_floor']}",
               rep_lpc["feat_var"] >= THRESH["feat_var_floor"], "FREEZE_PROTOCOL §4.4")
    c5 = pred("probe_saturation_met", 1.0 if saturated else 0.0, "==1", saturated, "FREEZE_PROTOCOL §4 saturation")
    not_collapse = c4a and c4b and c4c
    # collapse is the DOMINANT explanation: leakage dropped but task/structure ALSO collapsed
    collapse_dominant = (leak_gap > 0) and ((not not_collapse) or (util_loss > THRESH["util_loss_pp_max"]))
    if c1 and c2 and c3 and not_collapse and c5:
        decision = "RETAIN_LPC"
    elif collapse_dominant:
        decision = "DROP_LPC_COLLAPSE"
    else:
        decision = "INCONCLUSIVE"
    return decision, P

def saturation_met(tier_leak_kl):
    """§4 saturation: two CONSECUTIVE probe tiers improve attacker leakage by < delta_kl (and not still rising).
    tier_leak_kl ordered by increasing capacity (higher = more leakage detected)."""
    if len(tier_leak_kl) < 3:
        return False
    impr = [tier_leak_kl[i + 1] - tier_leak_kl[i] for i in range(len(tier_leak_kl) - 1)]
    return impr[-1] < THRESH["sat_delta_kl"] and impr[-2] < THRESH["sat_delta_kl"]

# ============================ MAIN ============================
def main():
    code_commit = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "nogit"
    ok_proto, proto_msg = verify_protocol_thresholds()
    frozen, manifest, gate_msg = freeze_gate()
    schema = schema_validate(DUMP_DIR)
    base = dict(code_commit=code_commit, protocol_check=proto_msg, freeze_gate=gate_msg,
                thresholds=THRESH, schema=schema)
    if not ok_proto:
        print(f"PROTOCOL THRESHOLD VERIFICATION FAILED: {proto_msg}"); sys.exit(3)

    if not frozen:
        # BLOCKED phase: implement+test+schema-validate; NO probe fitting on real data; NO decision file.
        os.makedirs("results/p15_audit_impl", exist_ok=True)
        rep = dict(status="BLOCKED_ON_FREEZE_A1", **base)
        json.dump(rep, open("results/p15_audit_impl/blocked_report.json", "w"), indent=2)
        print("=" * 70)
        print("STATUS: BLOCKED_ON_FREEZE_A1  (no Freeze A1 marker -> no real-data probe fitting, no decision file)")
        print(f"  protocol thresholds verified: {proto_msg}")
        print(f"  freeze gate: {gate_msg}")
        print(f"  feat_dump_v2: {schema['n_files']} files, keys={schema['present_keys']}")
        if schema["missing_for_full_audit"]:
            print("  METADATA GAPS for the full domain-leakage audit:")
            for g in schema["missing_for_full_audit"]:
                print(f"    - {g}")
        print("  implementation report -> results/p15_audit_impl/blocked_report.json")
        print("=" * 70)
        return  # clean terminal state (block is expected, not an error)

    # ---- FROZEN: gate the DOMAIN-LEAKAGE half on a VERIFIED metadata sidecar (group metadata) ----
    sidecar_ok, sidecar_msg = check_sidecar_bound(manifest)
    rep_partial = run_representation_half()              # always runnable from the dumps' TASK labels (no domain probe)
    if not sidecar_ok:
        # BLOCKED_ON_GROUP_METADATA: representation PARTIAL report only; NO retain/drop; NO p15_decision.json.
        os.makedirs("results/p15_audit_partial", exist_ok=True)
        json.dump(dict(status="BLOCKED_ON_GROUP_METADATA", sidecar=sidecar_msg,
                       representation_partial=rep_partial, **base),
                  open("results/p15_audit_partial/representation_partial.json", "w"), indent=2, default=str)
        print("STATUS: BLOCKED_ON_GROUP_METADATA")
        print(f"  freeze A1 present but sidecar NOT verified: {sidecar_msg}")
        print("  representation half -> results/p15_audit_partial/representation_partial.json (NO retain/drop, NO decision file)")
        return
    # ---- sidecar verified: full audit (atomic temp dir; rename only after every stage succeeds) ----
    out_final = f"results/p15_audit/{manifest['hash'][:16]}"
    if os.path.exists(out_final):
        print(f"output {out_final} exists — refusing to overwrite (immutable)"); sys.exit(4)
    tmp = tempfile.mkdtemp(prefix="p15_tmp_", dir="results")
    try:
        result = run_full_audit(manifest, rep_partial, base)   # domain-probe + representation + decision engine
        json.dump(result["decision"], open(f"{tmp}/p15_decision.json", "w"), indent=2, default=str)
        json.dump(result["manifest"], open(f"{tmp}/p15_manifest.json", "w"), indent=2, default=str)
        open(f"{tmp}/SHA256SUMS", "w").write(result["checksums"])
        os.rename(tmp, out_final)                         # atomic publish only after all stages succeed
        print(f"P1.5 decision: {result['decision']['decision']} -> {out_final}/p15_decision.json")
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"real-data audit aborted ({e}); no partial output left."); sys.exit(5)


def check_sidecar_bound(manifest):
    """Bound sidecar requires status==FEATURE_BOUND (both ERM+LPC branches) AND a matching Freeze A1 hash.
    REQUIRES_FROZEN_REDUMP / METADATA_BINDING_FAILED are reported verbatim (they are NOT scientific INCONCLUSIVE)."""
    p = "results/feat_dump_v2_sidecar/bound_manifest.json"
    if not os.path.exists(p):
        return False, "no bound sidecar (run p15_metadata_replay replay_and_bind post-Freeze-A1)"
    try:
        m = json.load(open(p))
    except Exception as e:
        return False, f"bound sidecar unreadable: {e}"
    if m.get("status") != "FEATURE_BOUND":
        return False, f"sidecar status={m.get('status')} (not FEATURE_BOUND): {m.get('detail')}"
    if not (m.get("erm_bound") and m.get("lpc_bound")):
        return False, f"sidecar not bound for BOTH branches (erm={m.get('erm_bound')}, lpc={m.get('lpc_bound')})"
    if m.get("freeze_a1_hash") != manifest.get("hash"):
        return False, "sidecar freeze hash != current Freeze A1 hash (version mismatch)"
    return True, "FEATURE_BOUND (erm+lpc) + version-matched"


def run_representation_half():
    """Representation-integrity + utility on feat_dump_v2 ERM vs LPC (TASK labels only; NO domain probe; NO retain/drop).
    Returns per-cohort metrics; this alone NEVER yields a retain/drop decision."""
    import re
    rng = np.random.default_rng(0); out = {}
    for f in sorted(glob.glob(f"{DUMP_DIR}/feat_*_erm_0.npz")):
        key = os.path.basename(f)[len("feat_"):-len("_erm_0.npz")]
        lpc = f.replace("_erm_0.npz", "_lpc_prior_0.3.npz")
        if not os.path.exists(lpc):
            continue
        oe, ol = np.load(f), np.load(lpc)
        out[key] = dict(erm=representation_metrics(oe["z_te"], oe["y_te"], np.random.default_rng(0)),
                        lpc=representation_metrics(ol["z_te"], ol["y_te"], np.random.default_rng(0)))
    return out


def _canon_hash_full(z):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(z, dtype="<f8")).tobytes()).hexdigest()

def run_full_audit(manifest, rep_partial, base):
    """Post-Freeze-A1 P1.5 audit consuming ONLY the hash-verified FEATURE_BOUND redump (feat_dump_v3). Verifies the
    bind-manifest hash AND re-hashes every shard's content (point 4); ANY mismatch -> hard stop, NO partial decision.
    Then grouped multi-capacity + alt-bias domain-leakage probes (D=source-cohort, grouped by cohort::subject,
    conditioned on Y), representation/utility metrics (z_te, erm vs lpc), and the pre-registered decision_engine."""
    from sklearn.preprocessing import LabelEncoder
    V3 = "results/feat_dump_v3"
    bm = json.load(open(f"{V3}/FEATURE_BOUND.json"))
    if bm.get("status") != "FEATURE_BOUND":
        raise RuntimeError(f"redump not FEATURE_BOUND ({bm.get('status')})")
    mh = bm.pop("manifest_hash")
    if hashlib.sha256(json.dumps(bm, sort_keys=True).encode()).hexdigest() != mh:
        raise RuntimeError("FEATURE_BOUND manifest hash mismatch")
    bm["manifest_hash"] = mh
    if bm["freeze_a1_hash"] != manifest["hash"]:
        raise RuntimeError("bind freeze hash != current Freeze A1 hash")
    rng = np.random.default_rng(0); ERM, LPC = "erm:0", "lpc_prior:0.3"
    leak = {ERM: defaultdict(list), LPC: defaultdict(list)}; rep = {ERM: [], LPC: []}
    for sh in bm["shards"]:
        fn = f"{V3}/audit_{sh['cond']}_{sh['cohort']}_{sh['config'].replace(':', '_')}.npz"
        o = np.load(fn, allow_pickle=True)
        if _canon_hash_full(o["z_te"]) != sh["feat_hash_te"] or _canon_hash_full(o["z_se"]) != sh["feat_hash_se"]:
            raise RuntimeError(f"CONTENT HASH MISMATCH {fn} — hard stop, no partial decision")
        cfg = sh["config"]; rep[cfg].append(representation_metrics(o["z_te"], o["y_te"], np.random.default_rng(0)))
        D = LabelEncoder().fit_transform([str(c) for c in o["cohort_id_se"]])
        if len(set(D)) < 2:
            continue
        Y = np.asarray(o["y_se"]); G = np.array([str(g) for g in o["group_id_se"]])
        Zin = np.c_[np.asarray(o["z_se"], float), np.eye(int(Y.max()) + 1)[Y]]      # condition on Y
        for t, v in domain_probe_audit(Zin, D, Y, G, PROBE_TIERS + PROBE_ALT, rng).items():
            leak[cfg][t].append(v["heldout_bacc"])
    m = lambda c, t: float(np.mean(leak[c][t])) if leak[c].get(t) else float("nan")
    rmk = lambda c, k: float(np.mean([r[k] for r in rep[c]])) if rep[c] else float("nan")
    strong = "mlp256x2"
    rep_erm = {k: rmk(ERM, k) for k in ("eff_rank", "stable_rank", "scatter_ratio", "feat_var", "task_bacc")}
    rep_lpc = {k: rmk(LPC, k) for k in ("eff_rank", "stable_rank", "scatter_ratio", "feat_var", "task_bacc")}
    tier_leak = [m(ERM, t) for t in PROBE_TIERS]
    decision, predicates = decision_engine(m(ERM, strong), m(LPC, strong), max(m(ERM, "gbt"), m(ERM, "knn")),
                                           max(m(LPC, "gbt"), m(LPC, "knn")), rep_erm, rep_lpc,
                                           saturation_met([abs(x) for x in tier_leak]))
    dec = dict(decision=decision, predicates=predicates, freeze_a1_hash=manifest["hash"],
               instrumentation_commit=bm["instrumentation_commit"], bind_manifest_hash=mh, n_shards=len(bm["shards"]),
               leakage=dict(strong_probe=strong, erm=m(ERM, strong), lpc=m(LPC, strong),
                            alt_erm=max(m(ERM, "gbt"), m(ERM, "knn")), alt_lpc=max(m(LPC, "gbt"), m(LPC, "knn")),
                            tier_ladder=dict(zip(PROBE_TIERS, tier_leak))),
               representation=dict(erm=rep_erm, lpc=rep_lpc))
    checks = "\n".join(f"{sh['cond']}/{sh['cohort']}/{sh['config']}  {sh['feat_hash_te'][:16]}" for sh in bm["shards"])
    return dict(decision=dec, manifest=dict(base, bind_manifest_hash=mh, V3=V3), checksums=checks)


if __name__ == "__main__":
    main()
