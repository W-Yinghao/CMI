#!/usr/bin/env python
"""FSR Phase 8C-1 — PhysioNetMI F1-spatial feature dump for ALL 105 analyzable subjects (one model). Runs 4/8/12
(imagined L/R fist), 64ch @160->200Hz, F1 = per-channel (mean over patches, keep channels -> 64*200=12800). Saves
X (F1), y (0=left/1=right), d (subject), run (4/8/12 for run-held-out L1). No target labels used (pure extraction).
GPU. Determinism per 8B. Writes physionetmi_<model>_F1.npz + manifest.

    <eeg2025 python> scripts/physio_8c1_dump.py --model cbramod
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_rq4/scripts")
import cb_cbm_feature_dump as FD
from physio_8c0_encoder_sanity import load_subject, preprocess

OUT = Path("results/fsr_codebrain_cbramod_8c")
EXCLUDE = [88, 89, 92, 100]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); args = ap.parse_args()
    (OUT / "embeddings").mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4); FD.set_determinism(torch)
    dev = torch.device("cuda")
    fwd, _, loadinfo = (FD.build_codebrain if args.model == "codebrain" else FD.build_cbramod)(torch, dev)
    subjects = [s for s in range(1, 110) if s not in EXCLUDE]

    def f1(Xr):
        x = preprocess(Xr); out = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                feats = fwd(torch.tensor(x[i:i + 64], device=dev)).cpu().numpy()
                out.append(feats.mean(axis=2).reshape(feats.shape[0], feats.shape[1] * feats.shape[3]))
        return np.concatenate(out, 0)

    X, Y, D, R = [], [], [], []
    for s in subjects:
        Xr, y, r = load_subject(s)
        X.append(f1(Xr).astype(np.float32)); Y.append(y); D.append(np.full(len(y), s)); R.append(r)
        print(f"[{args.model}] S{s:03d} n={len(y)} done", flush=True)
    X = np.concatenate(X, 0); Y = np.concatenate(Y); D = np.concatenate(D); R = np.concatenate(R)
    np.savez(OUT / "embeddings" / f"physionetmi_{args.model}_F1.npz", X=X, y=Y, d=D, run=R)
    (OUT / f"physionetmi_dump_manifest_{args.model}.json").write_text(json.dumps(dict(
        model=args.model, n_trials=int(len(Y)), n_subjects=int(np.unique(D).size), F1_dim=int(X.shape[1]),
        classes=sorted(int(c) for c in np.unique(Y)), runs=sorted(int(c) for c in np.unique(R)),
        load=loadinfo), indent=2) + "\n")
    print(f"WROTE physionetmi_{args.model}_F1.npz X={X.shape} subjects={np.unique(D).size}")


if __name__ == "__main__":
    main()
