#!/usr/bin/env python
"""FSR Phase 7B-0 — learnability/power GATE verdict (fail-closed). Pools the head_learnability_gate.csv over
(dataset,subject) folds x confirm seeds with a clustered bootstrap; decides whether the skewed head DID learn the
source subject->c_d shortcut (so 7B-1 target-harm/repair is interpretable). Reports the key DISSOCIATION between
cd-prediction-bias (prior-like) and subject-subspace reliance (representation)."""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_head_only_learned_reliance")
RNG = np.random.default_rng(0)
CONFIRM = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
RHO_HI = 0.8


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def clu_of(rows):
    seen, idx = {}, []
    for r in rows:
        idx.append(seen.setdefault((r["dataset"], r["target_subject"]), len(seen)))
    return np.array(idx)


def _res(clu, rng):
    u = np.unique(clu); pick = rng.choice(u, len(u), replace=True)
    return np.concatenate([np.where(clu == c)[0] for c in pick])


def boot_ci(v, clu, nb=3000):
    v = np.asarray(v, float)
    b = [v[_res(clu, RNG)].mean() for _ in range(nb)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main():
    rows = [r for r in csv.DictReader(open(R / "head_learnability_gate.csv")) if int(fl(r["token_seed"])) in CONFIRM]
    subj_dec = np.array([fl(r["subj_decode_bacc"]) for r in rows]); chance = np.array([fl(r["chance"]) for r in rows])
    subject_decodable = bool(np.mean(subj_dec - chance) > 0.20 and np.mean([r["subj_decodable"] == "True" for r in rows]) > 0.9)
    py_ok = bool(np.max([fl(r["py_match_max"]) for r in rows]) < 1e-6)   # reweighting holds P(y) exactly

    dose = {}
    for rho in ("0.0", "0.5", "0.8"):
        rr = [r for r in rows if r["rho"] == rho]; clu = clu_of(rr)
        cd = np.array([fl(r["cd_pred_rate_minus_H0"]) for r in rr])
        l5 = np.array([fl(r["l5_minus_H0"]) for r in rr])
        dose[rho] = dict(cd_bias_mean=round(float(cd.mean()), 4), cd_bias_ci=boot_ci(cd, clu),
                         l5_reliance_mean=round(float(l5.mean()), 4), l5_reliance_ci=boot_ci(l5, clu), n=len(rr))
    # monotone rising cd-bias + CI at rho=0.8 excludes 0 => head DID learn a subject->c_d (prior-like) shortcut
    cd_rise = bool(dose["0.8"]["cd_bias_mean"] > dose["0.5"]["cd_bias_mean"] > dose["0.0"]["cd_bias_mean"] and
                   dose["0.8"]["cd_bias_ci"][0] > 0)
    l5_rise = bool(dose["0.8"]["l5_reliance_ci"][0] > 0)   # representation-subspace reliance rose?
    learned_via_prediction_bias = cd_rise
    learned_via_subspace_reliance = l5_rise
    gate_pass = bool(subject_decodable and py_ok and (learned_via_prediction_bias or learned_via_subspace_reliance))

    if not subject_decodable:
        reason = "subject_not_head_decodable"
    elif not gate_pass:
        reason = "head_did_not_learn_shortcut_under_this_protocol"
    else:
        reason = ("learned_as_prediction_bias_not_subspace_reliance" if (cd_rise and not l5_rise)
                  else "learned_shortcut")

    verdict = dict(
        stage="7B-0_learnability_gate", subject_decodable=subject_decodable,
        subject_decode_bacc_mean=round(float(subj_dec.mean()), 4), chance_mean=round(float(chance.mean()), 4),
        py_held_exactly=py_ok, eff_n_frac=1.0,
        dose_response=dose,
        learned_via_prediction_bias=learned_via_prediction_bias,
        learned_via_subspace_reliance=learned_via_subspace_reliance,
        dissociation=bool(cd_rise and not l5_rise),
        gate_pass=gate_pass, gate_reason=reason,
        proceed_to_7B1=gate_pass,
        note=("DISSOCIATION: the head learns a subject->c_d PREDICTION BIAS (prior-like) that rises with rho, but "
              "does NOT increase reliance on the subject REPRESENTATION subspace -> the learned reliance is a "
              "prior/decision-bias shortcut (recoverability R3-prior), not a subject-subspace shortcut. If gate "
              "passes, 7B-1 tests target harm + whether E4/E4b/erasure (target-X) or prior-directed repair helps."
              if (cd_rise and not l5_rise) else
              "gate outcome; see reason. 7B-1 runs only if gate_pass."),
    )
    (R / "head_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    interp = [
        dict(gate="subject_decodable", status=str(subject_decodable),
             interpretation=f"linear subject decode {round(float(subj_dec.mean()),3)} vs chance {round(float(chance.mean()),3)}",
             allowed_claim="subject signal is (linearly) decodable from the frozen concat latents",
             forbidden_claim="subject signal is absent", pc2_implication="n/a"),
        dict(gate="learned_via_prediction_bias", status=str(cd_rise),
             interpretation=f"cd-pred-bias vs rho: {dose['0.0']['cd_bias_mean']}/{dose['0.5']['cd_bias_mean']}/{dose['0.8']['cd_bias_mean']} (CI@.8 {dose['0.8']['cd_bias_ci']})",
             allowed_claim="a source subject-class skew induces a subject-conditional PREDICTION BIAS in the head" if cd_rise else "no head-level subject->c_d prediction bias induced",
             forbidden_claim="the head weaponizes the subject representation" if cd_rise else "natural subject signal is safe",
             pc2_implication="prior-like reliance -> prior alignment, not subject erasure" if cd_rise else "head-only skew does not weaponize -> PC2 low value"),
        dict(gate="learned_via_subspace_reliance", status=str(l5_rise),
             interpretation=f"l5 subject-subspace reliance vs rho: {dose['0.0']['l5_reliance_mean']}/{dose['0.5']['l5_reliance_mean']}/{dose['0.8']['l5_reliance_mean']} (CI@.8 {dose['0.8']['l5_reliance_ci']})",
             allowed_claim="the head increases subject-subspace reliance under skew" if l5_rise else "the head does NOT increase subject-subspace reliance under skew (representation not weaponized)",
             forbidden_claim="n/a", pc2_implication="representation shortcut" if l5_rise else "not a representation shortcut"),
        dict(gate="gate_pass", status=str(gate_pass),
             interpretation=reason,
             allowed_claim="7B-1 target-harm/repair is interpretable" if gate_pass else "7B-0 fail-closed: do NOT run 7B-1 target-harm/repair",
             forbidden_claim="natural EEG subject signal can never be harmful",
             pc2_implication="see FSR_38 Q8 (>=3 datasets + PM go required regardless)"),
    ]
    with open(R / "head_result_interpretation_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(interp[0].keys())); w.writeheader()
        for r in interp:
            w.writerow(r)

    print("Phase 7B-0 learnability gate:")
    print(f"  subject_decodable={subject_decodable} (decode {round(float(subj_dec.mean()),3)} vs chance {round(float(chance.mean()),3)}); P(y) held exactly={py_ok}")
    for rho in ("0.0", "0.5", "0.8"):
        d = dose[rho]
        print(f"  rho={rho}: cd_pred_bias(H1-H0)={d['cd_bias_mean']} ci={d['cd_bias_ci']}  |  l5_subspace_reliance(H1-H0)={d['l5_reliance_mean']} ci={d['l5_reliance_ci']}")
    print(f"  learned_via_prediction_bias={learned_via_prediction_bias}  learned_via_subspace_reliance={learned_via_subspace_reliance}  DISSOCIATION={verdict['dissociation']}")
    print(f"  ==> gate_pass={gate_pass}  reason={reason}  proceed_to_7B1={gate_pass}")


if __name__ == "__main__":
    main()
