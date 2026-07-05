"""C11c — SRC one-fold pilot (BNCI2014-001, one target/seed, levels 0+1). Method-polishing pilot, NOT a
confirmatory conclusion. Trains ERM + OACI (negative reference) + SRC via the REAL engine (train_stage1 +
train_stage2 — not a shortcut loop), reusing FoldData / level context / leakage measurement / eval metrics /
target-isolation provenance. Each method is selected by ITS OWN rule (ERM = stage1; OACI = min selection
leakage point; SRC = source_endpoint_selector over source_guard). Then the 3 selected checkpoints are
evaluated on source_audit + target for the K2 worst-domain endpoints; K1 audit leakage is computed as pure
MEASUREMENT (never drives SRC selection). Target is EVAL-only; a RunProvenance asserts target_fit_ids empty.

Writes oaci/reports/C11_SRC_ONEFOLD_PILOT.{md,json}. NO artifact tree (that path is 4-method-locked); the
pilot report is canonical-hashed + round-trip verifiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from ..artifacts.canonical_json import canonical_json_bytes
from ..diagnostics.candidate_replay import _PREFIX, _ROLES, _bundle, _leak_point
from ..eval.calibration import fixed_bin_edges
from ..methods.source_robust import SRCObjective
from ..runner.audit import build_training_data_for_design
from ..runner.bnci_data import build_bnci_real_fold
from ..runner.metrics import evaluate_prediction_bundle
from ..runner.objectives import make_objective
from ..runner.provenance import RunnerPhase, RunProvenance
from ..runner.selection import unique_feasible_records
from ..runner.source_endpoint_selector import select_source_endpoint
from ..runner.stage1 import run_stage1_once
from ..runner.staged_fold import _level_contexts
from ..train.engine import InvocationRegistry, train_stage2
from .loso_plan import explicit_split as _loso_split, loso_fold_spec
from .materialize import VALIDATION_BOOTSTRAP, materialize_pilot_manifest
from .schema import load_confirmatory


def _materialize(protocol_path, dataset, target, seed, manifest_out, bootstrap_mode):
    proto = load_confirmatory(protocol_path)
    override = VALIDATION_BOOTSTRAP if bootstrap_mode == "validation" else None
    spec = loso_fold_spec(int(target), dataset_id=str(dataset))
    return materialize_pilot_manifest(proto, dataset, target_subject=int(target), out_path=manifest_out,
                                      model_seeds=[int(seed)], bootstrap_override=override,
                                      explicit_split=_loso_split(spec), deleted_cell=dict(spec["deleted_cell"]))


def _audit_leak(fold, ss, rk, level, mh, ms, device):
    sa = fold.fold_scope.source_audit
    if sa.status != "estimable" or sa.fold_plan is None:
        return None
    atd = build_training_data_for_design(fold.fold_data, sa.design)
    return _leak_point(ms, mh, atd, sa.design, sa.support_graph, sa.fold_plan, "audit_feature_factory",
                       (mh,), fold, rk, device)


def _sel_leak(fold, ss, lp, plans, rk, level, mh, ms, device):
    if plans.selection_status != "estimable" or plans.selection_fold_plan is None:
        return None
    return _leak_point(ms, mh, lp.training_data, plans.selection_design, ss.support_graph,
                       plans.selection_fold_plan, "selection_feature", (), fold, rk, device)


def _metrics(fold, ss, rk, level, views, dmap, edges, name, mh, ms, device, roles=_ROLES):
    out = {}
    for role in roles:
        b = _bundle(ms, mh, name, role, views[role], dmap[role], rk, fold, ss, level, device)
        m = evaluate_prediction_bundle(b, bin_edges=edges)
        pre = _PREFIX[role]
        out[f"{pre}_worst_bacc"] = m.worst_domain_reference_bacc
        out[f"{pre}_worst_nll"] = m.worst_domain_nll
        out[f"{pre}_worst_ece"] = m.worst_domain_ece
    return out


def run_src_onefold(protocol_path, dataset, target, seed, datalake_root, manifest_out, device, *,
                    bootstrap_mode="validation", smooth_temperature=0.1, margins=None):
    _mpath, _manifest = _materialize(protocol_path, dataset, target, seed, manifest_out, bootstrap_mode)
    fold = build_bnci_real_fold(_mpath, datalake_root)
    fold.fold_data.assert_integrity()
    fd, maps, fs = fold.fold_data, fold.maps, fold.fold_scope
    exec_cfg, model_spec, mfac = fold.execution_config, fold.model_spec, fold.model_factory()
    nc = len(maps.class_names)
    tol = float(exec_cfg.engine_template.numerical_tol)
    dmap = {"source_guard": maps.source_domain_to_index, "source_audit": maps.evaluation_domain_to_index,
            "target_audit": maps.evaluation_domain_to_index}
    levels_out = {}
    for level, rk, ss, lp, plans in _level_contexts(fold, int(seed), str(dataset)):
        engine_cfg = exec_cfg.engine_config_for(rk)
        stage1 = run_stage1_once(rk, lp, plans, mfac, model_spec, engine_cfg,
                                 exec_cfg.execution_config_hash, InvocationRegistry(), device)
        data = lp.training_data
        tau = float(stage1.erm_stage.tau)
        edges = fixed_bin_edges(exec_cfg.ece_bins)
        views = {r: fd.make_role_view("source_guard", ss.source_train_idx) if r == "source_guard"
                 else fd.make_role_view(r) for r in _ROLES}

        # ---- target-isolation provenance (reuse the runner's asserts) ----
        prov = RunProvenance()
        prov.record_fit("preprocess", fd.preprocess_fit_ids)
        prov.transition(RunnerPhase.TRAINING)
        prov.record_fit("optimization", ss.source_train_sample_ids)
        prov.transition(RunnerPhase.SELECTION)

        # ---- train ERM (stage1) + OACI + SRC via the engine ----
        erm_ckpt = stage1.erm_stage.checkpoint
        oaci_obj, _ = make_objective("OACI", ss, fs, exec_cfg)
        oaci_tr = train_stage2(mfac, stage1.erm_stage, data, oaci_obj, plans.stage2_task,
                               plans.oaci_alignment, engine_cfg, device)
        n_src_dom = int(torch.unique(data.d).numel())
        src_obj = SRCObjective(nc, n_src_dom, smooth_temperature=smooth_temperature)
        src_tr = train_stage2(mfac, stage1.erm_stage, data, src_obj, plans.stage2_task,
                              plans.full_domain_alignment, engine_cfg, device)

        # ---- ERM reference metrics (source_guard needed for the SRC guard) ----
        erm_m = _metrics(fold, ss, rk, level, views, dmap, edges, "ERM", erm_ckpt.model_hash,
                         erm_ckpt.model_state, device)
        erm_al = _audit_leak(fold, ss, rk, level, erm_ckpt.model_hash, erm_ckpt.model_state, device)

        # ---- OACI selection: min selection-leakage point over feasible candidates (OACI's own objective) ----
        oaci_feas = unique_feasible_records(oaci_tr, numerical_tol=tol)
        oaci_best, oaci_best_leak = erm_ckpt, _sel_leak(fold, ss, lp, plans, rk, level, erm_ckpt.model_hash,
                                                        erm_ckpt.model_state, device)
        for rec in oaci_feas:
            sl = _sel_leak(fold, ss, lp, plans, rk, level, rec.model_hash, rec.model_state, device)
            if sl is not None and (oaci_best_leak is None or sl < oaci_best_leak):
                oaci_best, oaci_best_leak = rec, sl

        # ---- SRC selection: source_endpoint_selector over source_guard metrics (source-only) ----
        src_table = [{"model_hash": erm_ckpt.model_hash, "is_erm": True, "feasible": True, "R_src": erm_ckpt.R_src,
                      "source_guard_worst_bacc": erm_m["source_guard_worst_bacc"],
                      "source_guard_worst_nll": erm_m["source_guard_worst_nll"],
                      "source_guard_worst_ece": erm_m["source_guard_worst_ece"]}]
        src_feas = unique_feasible_records(src_tr, numerical_tol=tol)
        src_state = {erm_ckpt.model_hash: erm_ckpt.model_state}
        for rec in src_feas:
            sm = _metrics(fold, ss, rk, level, views, dmap, edges, "SRC", rec.model_hash, rec.model_state,
                          device, roles=("source_guard",))
            src_table.append({"model_hash": rec.model_hash, "is_erm": False, "feasible": True, "R_src": rec.R_src,
                              "source_guard_worst_bacc": sm["source_guard_worst_bacc"],
                              "source_guard_worst_nll": sm["source_guard_worst_nll"],
                              "source_guard_worst_ece": sm["source_guard_worst_ece"]})
            src_state[rec.model_hash] = rec.model_state
        src_sel = select_source_endpoint(src_table, tau, margins=margins, tol=tol)
        prov.record_fit("selection", ss.source_train_sample_ids)      # SRC/OACI selection touch source only

        # ---- evaluate the three SELECTED checkpoints on source_audit + target ----
        def full(name, mh, ms):
            m = _metrics(fold, ss, rk, level, views, dmap, edges, name, mh, ms, device)
            m["audit_leakage_point"] = _audit_leak(fold, ss, rk, level, mh, ms, device)
            m["model_hash"] = mh
            return m
        erm_full = {**erm_m, "audit_leakage_point": erm_al, "model_hash": erm_ckpt.model_hash}
        oaci_full = full("OACI", oaci_best.model_hash, oaci_best.model_state)
        src_sel_mh = src_sel["chosen_model_hash"]
        src_full = full("SRC", src_sel_mh, src_state[src_sel_mh])

        # ---- K2 (target worst-domain) + K1 (audit leakage, MEASUREMENT) deltas vs ERM ----
        def d(a, b):
            return None if (a is None or b is None) else float(a) - float(b)
        def deltas(sel):
            return {"K2_delta_target_worst_bacc": d(sel["target_worst_bacc"], erm_full["target_worst_bacc"]),
                    "K2_delta_target_worst_nll": d(sel["target_worst_nll"], erm_full["target_worst_nll"]),
                    "K2_delta_target_worst_ece": d(sel["target_worst_ece"], erm_full["target_worst_ece"]),
                    "K2_delta_source_audit_worst_bacc": d(sel["source_audit_worst_bacc"], erm_full["source_audit_worst_bacc"]),
                    "K1_delta_audit_leakage_MEASUREMENT_ONLY": d(sel["audit_leakage_point"], erm_full["audit_leakage_point"])}

        prov.transition(RunnerPhase.SELECTION_LOCKED)
        prov.transition(RunnerPhase.AUDIT)
        prov.transition(RunnerPhase.COMPLETE)
        levels_out[str(level)] = {
            "n_source_domains": n_src_dom, "tau": tau, "smooth_temperature": smooth_temperature,
            "ERM": erm_full, "OACI": {**oaci_full, "selected_R_src": float(oaci_best.R_src),
                                      "risk_feasible": bool(oaci_best.R_src <= tau + tol), **deltas(oaci_full)},
            "SRC": {**src_full, "selected_R_src": float(_rsrc(src_state, src_feas, erm_ckpt, src_sel_mh)),
                    "risk_feasible": bool(_rsrc(src_state, src_feas, erm_ckpt, src_sel_mh) <= tau + tol),
                    "fallback_erm": src_sel["fallback_erm"], "selection_reason": src_sel["selection_reason"],
                    "n_feasible": src_sel["n_feasible"], "n_guard_pass": src_sel["n_guard_pass"],
                    "access": src_sel["access"], **deltas(src_full)},
            "target_fit_ids_empty": not prov.snapshot().target_fit_ids,
            "src_selector_target_read": src_sel["access"]["target_read"]}
    return _assemble(dataset, target, seed, smooth_temperature, levels_out)


def _rsrc(src_state, src_feas, erm_ckpt, mh):
    if mh == erm_ckpt.model_hash:
        return float(erm_ckpt.R_src)
    for rec in src_feas:
        if rec.model_hash == mh:
            return float(rec.R_src)
    return float(erm_ckpt.R_src)


def _assemble(dataset, target, seed, temp, levels_out):
    body = {"pilot": "C11c_SRC_onefold", "dataset": str(dataset), "target": int(target), "seed": int(seed),
            "smooth_temperature": temp, "methods": ["ERM", "OACI", "SRC"], "levels": levels_out,
            "notice": "method-polishing pilot; NOT a confirmatory conclusion. K1 leakage is MEASUREMENT ONLY."}
    body["all_target_fit_ids_empty"] = all(lv["target_fit_ids_empty"] for lv in levels_out.values())
    body["no_selector_read_target"] = all(not lv["src_selector_target_read"] for lv in levels_out.values())
    body["pilot_hash"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def deep_verify_pilot(body) -> bool:
    """Round-trip / hash verification of the pilot report (the pilot's 'deep verify')."""
    b = {k: v for k, v in body.items() if k != "pilot_hash"}
    return body.get("pilot_hash") == hashlib.sha256(canonical_json_bytes(b)).hexdigest()


def _signal(body) -> dict:
    """Answer the C11c questions from the per-level SRC deltas."""
    lv = body["levels"]
    def worst(metric):
        return [lv[L]["SRC"][metric] for L in lv if lv[L]["SRC"][metric] is not None]
    tb = worst("K2_delta_target_worst_bacc"); tn = worst("K2_delta_target_worst_nll")
    feas = all(lv[L]["SRC"]["risk_feasible"] for L in lv)
    improves_bacc = bool(tb) and all(x > 0 for x in tb)
    not_worse_nll = bool(tn) and all(x <= 0 for x in tn)
    n_fallback = sum(1 for L in lv if lv[L]["SRC"]["fallback_erm"])
    signal = feas and (improves_bacc or not_worse_nll) and n_fallback < len(lv)
    return {"risk_feasible_all_levels": feas, "target_worst_bacc_improves": improves_bacc,
            "target_worst_nll_not_worse": not_worse_nll, "src_fallback_levels": n_fallback,
            "SRC_shows_signal": signal,
            "verdict": ("SRC shows a one-fold signal -> proceed to BNCI001 LOSO seeds[0,1,2] with SRC"
                        if signal else "SRC shows NO one-fold signal here -> source-only endpoint optimization "
                        "does not transfer under this protocol; consider measurement-only / negative-result "
                        "direction rather than another DG penalty")}


def render_md(body) -> str:
    s = _signal(body)
    L = [f"# C11c — SRC one-fold pilot ({body['dataset']} target-{body['target']:03d} seed-{body['seed']})", "",
         f"> {body['notice']}", "",
         f"- target_fit_ids empty (all levels): **{body['all_target_fit_ids_empty']}** · SRC selector read "
         f"target: **{body['no_selector_read_target'] is False}** (must be False) · pilot deep-verify: "
         f"**{deep_verify_pilot(body)}**", "",
         "| level | method | tgt worst bAcc | tgt worst NLL | ΔK2 bAcc(vsERM) | ΔK2 NLL | ΔK1 leak(meas) | risk-feasible |",
         "|---:|---|---:|---:|---:|---:|---:|---|"]
    for lk, lv in body["levels"].items():
        L.append(f"| {lk} | ERM | {_f(lv['ERM']['target_worst_bacc'])} | {_f(lv['ERM']['target_worst_nll'])} | — | — | — | — |")
        for mth in ("OACI", "SRC"):
            m = lv[mth]
            L.append(f"| {lk} | {mth} | {_f(m['target_worst_bacc'])} | {_f(m['target_worst_nll'])} | "
                     f"{_f(m['K2_delta_target_worst_bacc'])} | {_f(m['K2_delta_target_worst_nll'])} | "
                     f"{_f(m['K1_delta_audit_leakage_MEASUREMENT_ONLY'])} | {m['risk_feasible']} |")
    L += ["", "## SRC selector behaviour", ""]
    for lk, lv in body["levels"].items():
        m = lv["SRC"]
        L.append(f"- level {lk}: reason={m['selection_reason']}, fallback_ERM={m['fallback_erm']}, "
                 f"feasible={m['n_feasible']}, guard_pass={m['n_guard_pass']}, "
                 f"selector roles read={m['access']['roles_actually_read']}, target_read={m['access']['target_read']}")
    L += ["", "## C11c signal", "",
          f"- risk-feasible all levels: **{s['risk_feasible_all_levels']}**",
          f"- target worst bAcc improves (all levels): **{s['target_worst_bacc_improves']}**",
          f"- target worst NLL not worse (all levels): **{s['target_worst_nll_not_worse']}**",
          f"- SRC fell back to ERM in **{s['src_fallback_levels']}/{len(body['levels'])}** levels",
          f"- **SRC shows signal: `{s['SRC_shows_signal']}`**", "", f"> {s['verdict']}"]
    return "\n".join(L)


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.src_onefold")
    ap.add_argument("--protocol", default="oaci/protocol/confirmatory_v2.yaml")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datalake-root", default="/projects/EEG-foundation-model/datalake/raw")
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--bootstrap-mode", default="validation")
    ap.add_argument("--smooth-temperature", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.device:
        device = args.device
    else:
        from ..runtime.cuda import configure_cuda_determinism
        device, _ = configure_cuda_determinism()
    body = run_src_onefold(args.protocol, args.dataset, args.target, args.seed, args.datalake_root,
                           args.manifest_out, device, bootstrap_mode=args.bootstrap_mode,
                           smooth_temperature=args.smooth_temperature)
    for p in (args.out_md, args.out_json):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes({**body, "signal": _signal(body)}))
    with open(args.out_md, "w") as f:
        f.write(render_md(body))
    s = _signal(body)
    print(f"wrote {args.out_json} + {args.out_md}: SRC_shows_signal={s['SRC_shows_signal']}, "
          f"target_fit_ids_empty={body['all_target_fit_ids_empty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
