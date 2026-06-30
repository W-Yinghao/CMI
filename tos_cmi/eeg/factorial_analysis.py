"""Track C analysis -- removability vs (architecture, latent dimension). Disentangles representation TYPE
(SPD/TSMNet vs conv/EEGNet) from latent DIMENSION in the "low-rank removability is representation-dependent"
finding. For each factorial cell (backbone x dim), runs the erasure/deletion probes (reusing
erasure_baselines.analyze) over the LOSO folds and reports the key removability signals:
  subject decode after LEACE (linear & MLP), after TOS V_D, random-k; task decode; selectivity; nDcand.
KEY question: does the nonlinear (MLP) subject residual after optimal linear erasure rise with latent
DIMENSION regardless of architecture (=> capacity-mediated), or split by architecture (=> type)?
  python -m tos_cmi.eeg.factorial_analysis [seed]
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys
import numpy as np
from tos_cmi.eeg.erasure_baselines import analyze

FACT = "tos_cmi/results/tos_cmi_eeg_frozen/factorial"
# include the already-existing full-dim cells (210 / 16) from the main LOSO dirs
EXTRA = {"TSMNet_dim20": "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO",
         "EEGNet_dim16": "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_EEGNet_LOSO"}


def _cells():
    out = {}
    for d in sorted(glob.glob("%s/BNCI2014_001_*_dim*" % FACT)):
        m = re.search(r"_(TSMNet|EEGNet)_dim(\d+)$", d)
        if m:
            out["%s_dim%s" % (m.group(1), m.group(2))] = d
    out.update(EXTRA)
    return out


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "0"
    rows = []
    for cell, d in sorted(_cells().items()):
        bb = "TSMNet" if "TSMNet" in cell else "EEGNet"
        m = float(np.sqrt(2 * int(cell.split("dim")[1]) + 0.25) - 0.5) if bb == "TSMNet" else None  # tangent->m (info only)
        paths = sorted(glob.glob("%s/sub*_erm_lam0_seed%s.npz" % (d, seed)))
        if not paths:
            continue
        rr = []
        for p in paths:
            try:
                rr.append(analyze(p))
            except Exception as e:
                print("[FAIL] %s : %r" % (p.split("/")[-1], e), flush=True)
        if not rr:
            continue
        agg = lambda k: float(np.nanmean([r[k] for r in rr if k in r]))
        z = rr[0]["z_dim"]
        rows.append({"cell": cell, "backbone": bb, "z_dim": z, "n_folds": len(rr), "nDcand": agg("nDcand"),
                     "chance_subj": rr[0]["chance_subj"],
                     "subj_full_mlp": agg("subj_full_mlp"),
                     "subj_LEACE_lin": agg("subj_LEACE_lin"), "subj_LEACE_mlp": agg("subj_LEACE_mlp"),
                     "subj_TOS_mlp": agg("subj_TOS_VD_mlp"), "subj_rand_mlp": agg("subj_random_k_mlp"),
                     "task_full_lin": agg("task_full_lin"), "task_LEACE_lin": agg("task_LEACE_lin"),
                     "task_TOS_lin": agg("task_TOS_VD_lin")})
        print("[%s] z=%d folds=%d | full_mlp=%.2f LEACE lin=%.2f mlp=%.2f | TOS_mlp=%.2f rand=%.2f | task full=%.2f LEACE=%.2f"
              % (cell, z, len(rr), rows[-1]["subj_full_mlp"], rows[-1]["subj_LEACE_lin"], rows[-1]["subj_LEACE_mlp"],
                 rows[-1]["subj_TOS_mlp"], rows[-1]["subj_rand_mlp"], rows[-1]["task_full_lin"], rows[-1]["task_LEACE_lin"]),
              flush=True)
    rows.sort(key=lambda r: (r["backbone"], r["z_dim"]))
    print("\n=== removability vs (architecture, dimension) [LEACE = optimal linear erasure] ===")
    print("%-14s z_dim | full_mlp  LEACE_lin LEACE_mlp(residual) TOS_mlp  rand_mlp | task_full task_LEACE  nDcand" % "backbone")
    for r in rows:
        print("%-14s %5d | %.3f     %.3f     %.3f              %.3f    %.3f    | %.3f     %.3f      %.1f"
              % (r["backbone"], r["z_dim"], r["subj_full_mlp"], r["subj_LEACE_lin"], r["subj_LEACE_mlp"],
                 r["subj_TOS_mlp"], r["subj_rand_mlp"], r["task_full_lin"], r["task_LEACE_lin"], r["nDcand"]))
    print("\nchance subject decode = %.3f" % (rows[0]["chance_subj"] if rows else float("nan")))
    print("READ: if LEACE_mlp residual RISES with z_dim within BOTH archs -> capacity-mediated; "
          "if it stays low for conv / high for SPD at matched z_dim -> architecture-mediated.")
    os.makedirs(FACT, exist_ok=True)
    json.dump(rows, open("%s/factorial_removability_seed%s.json" % (FACT, seed), "w"), indent=1)
    print("FACTORIAL_ANALYSIS_DONE (%d cells)" % len(rows))


if __name__ == "__main__":
    main()
