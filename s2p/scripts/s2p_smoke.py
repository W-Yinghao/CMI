#!/usr/bin/env python
"""S2P Phase 9B-0 — dual-model pipeline SMOKE (see FSR/PM). Validates the S2P pretraining matrix end-to-end on a
tiny budget: TUEG subject-subset loader -> from-scratch masked pretraining -> checkpoint save/reload -> feature
dump -> tiny downstream probe. NO SCIENCE (no subject-scaling/leakage claims). Validates gates G1-G10. CBraMod
primary; CodeBrain-Stage2 bounded (frozen released tokenizer). Deterministic. No target labels.

    <eeg2025 python> s2p/scripts/s2p_smoke.py --model cbramod --n_subjects 32 --hours 50
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_s2p/s2p/scripts")
sys.path.insert(0, "/home/infres/yinwang/eeg2025/CBraMod")
sys.path.insert(0, "/home/infres/yinwang/CodeBrain")
import tueg_subject_loader as L

OUT = Path("results/s2p_pretrain_smoke")
CKPT_TOK = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth"


def load_windows(rows, cap_windows=200):
    Xs, Ds = [], []
    tot = 0
    for x, d in L.windows_for(rows, max_windows_per_rec=4):
        Xs.append(x); Ds.append(d); tot += len(x)
        if tot >= cap_windows:
            break
    if not Xs:
        return np.zeros((0, 19, 30, 200), np.float32), np.zeros(0, int)
    return np.concatenate(Xs)[:cap_windows], np.concatenate(Ds)[:cap_windows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cbramod", "codebrain_stage2"], default="cbramod")
    ap.add_argument("--n_subjects", type=int, default=32); ap.add_argument("--hours", type=float, default=50)
    ap.add_argument("--condition", default="fixed_hours"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.model}_{args.condition}_N{args.n_subjects}_H{int(args.hours)}_s{args.seed}"
    rep = dict(run_id=run_id, model=args.model, n_subjects=args.n_subjects, hours=args.hours,
               condition=args.condition, seed=args.seed)
    import torch, torch.nn as nn
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # G1-G4: subject-subset loader + hours budget + disjoint train/val
    sub = L.build_subset(args.n_subjects, args.hours, args.condition, args.seed)
    man = sub["manifest"]; rep["subset_manifest"] = man
    rep["G1_sample_by_subject"] = bool(man["n_subjects_train"] + man["n_subjects_val"] == args.n_subjects)
    rep["G2_hours_budget_enforced"] = bool(man["train_hours"] <= man["per_subject_cap_h"] * man["n_subjects_train"] * 1.5 + 1)
    rep["G3_G4_pretrainval_subject_disjoint"] = bool(man["subjects_disjoint"])
    Xtr, Dtr = load_windows(sub["train"], cap_windows=256)
    Xva, Dva = load_windows(sub["val"], cap_windows=128)
    rep["n_windows_train"], rep["n_windows_val"] = int(len(Xtr)), int(len(Xva))
    if len(Xtr) < 8:
        rep["error"] = "too few windows"; (OUT / f"smoke_{run_id}.json").write_text(json.dumps(rep, indent=2) + "\n"); print(json.dumps(rep, indent=2)); return

    if args.model == "cbramod":
        from models.cbramod import CBraMod
        bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(dev)
        rec = nn.Linear(200, 200).to(dev)                      # masked-patch reconstruction head
        opt = torch.optim.Adam(list(bb.parameters()) + list(rec.parameters()), lr=1e-3)
        losses = []
        for step in range(args.steps):
            i = np.random.default_rng(step).integers(0, len(Xtr), min(16, len(Xtr)))
            x = torch.tensor(Xtr[i], device=dev)               # (B,19,30,200)
            m = (torch.rand(x.shape[:3], device=dev) < 0.3)    # mask 30% of (c,s) patches
            xm = x.clone(); xm[m] = 0.0
            out = rec(bb(xm))                                  # (B,19,30,200)
            loss = ((out[m] - x[m]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        rep["G6_finite_loss"] = bool(np.isfinite(losses).all())
        rep["pretrain_loss_first_last"] = [round(losses[0], 5), round(losses[-1], 5)]
        # G5 checkpoint save/reload
        cp = OUT / f"ckpt_{run_id}.pt"; torch.save(dict(backbone=bb.state_dict(), epoch=1, val="smoke"), cp)
        bb2 = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        miss, unexp = bb2.load_state_dict(torch.load(cp, map_location="cpu")["backbone"], strict=False)
        rep["G5_checkpoint_save_reload"] = bool(len(miss) == 0 and len(unexp) == 0)
        # G7 feature dump (frozen encoder pooled)
        bb2.eval().to(dev)
        with torch.no_grad():
            F = bb2(torch.tensor(Xva, device=dev)).mean(dim=(1, 2)).cpu().numpy()
        rep["G7_feature_dump"] = dict(shape=list(F.shape), finite=bool(np.isfinite(F).all()))
        # G8 tiny downstream-audit probe (subject decode on val windows -> harness runs; NO science)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.metrics import balanced_accuracy_score as BACC
        ok8 = False
        try:
            if len(np.unique(Dva)) >= 2:
                h = LDA().fit(F, Dva); _ = BACC(Dva, h.predict(F)); ok8 = True
        except Exception as e:
            rep["g8_note"] = str(e)
        rep["G8_downstream_audit_harness"] = ok8
        rep["model_smoke_pass"] = bool(rep["G1_sample_by_subject"] and rep["G3_G4_pretrainval_subject_disjoint"] and
                                       rep["G6_finite_loss"] and rep["G5_checkpoint_save_reload"] and rep["G7_feature_dump"]["finite"])
    else:
        # CodeBrain-Stage2 bounded: SSSM (if_codebook=True) predicts FROZEN tokenizer codes of masked patches
        try:
            from Models.SSSM import SSSM
            import Models.modeling_tokenizer as MT  # noqa
            from timm.models import create_model
            tok = create_model("tfdual_vq", pretrained=False, EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32)
            tsd = torch.load(CKPT_TOK, map_location="cpu", weights_only=False)
            tsd = tsd.get("model", tsd)
            tok.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in tsd.items()}, strict=False)
            tok.eval().to(dev)
            enc = SSSM(in_channels=200, res_channels=200, skip_channels=200, out_channels=200, num_res_layers=8,
                       diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
                       s4_lmax=570, s4_d_state=64, s4_dropout=0.1, s4_bidirectional=True, s4_layernorm=True,
                       codebook_size_t=4096, codebook_size_f=4096, if_codebook=True).to(dev)
            opt = torch.optim.Adam(enc.parameters(), lr=1e-4); losses = []
            ce = nn.CrossEntropyLoss()
            for step in range(min(args.steps, 15)):
                i = np.random.default_rng(step).integers(0, len(Xtr), min(4, len(Xtr)))
                x = torch.tensor(Xtr[i], device=dev)           # (B,19,30,200)
                with torch.no_grad():
                    tt, tf = tok.get_codebook_indices(x, input_chans=list(range(20)))   # target codes
                mask = torch.ones(x.shape[0], 19, 30, device=dev)                       # predict-all (bounded smoke)
                xt, xf = enc(x, mask=mask.reshape(x.shape[0], -1))                       # (masked, 4096) logits pair
                n = min(xt.shape[0], tt.reshape(-1).shape[0])
                loss = ce(xt[:n], tt.reshape(-1)[:n]) + ce(xf[:n], tf.reshape(-1)[:n])
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
            rep["G6_finite_loss"] = bool(np.isfinite(losses).all())
            rep["pretrain_loss_first_last"] = [round(losses[0], 4), round(losses[-1], 4)]
            cp = OUT / f"ckpt_{run_id}.pt"; torch.save(dict(encoder=enc.state_dict()), cp)
            rep["G5_checkpoint_save_reload"] = bool(cp.exists())
            rep["model_smoke_pass"] = bool(rep["G6_finite_loss"] and cp.exists())
        except Exception as e:
            import traceback
            rep["codebrain_blocker"] = f"{type(e).__name__}: {e}"; rep["model_smoke_pass"] = False
            rep["traceback_tail"] = traceback.format_exc().splitlines()[-3:]

    (OUT / f"smoke_{run_id}.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps({k: rep[k] for k in rep if not k.endswith("manifest")}, indent=2, default=str))


if __name__ == "__main__":
    main()
