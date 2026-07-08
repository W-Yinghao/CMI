#!/usr/bin/env python
"""S2P CodeBrain-Stage2 NATIVE-PATH smoke test (infra only; NO science claim, NO P1 inclusion).

Confirms the CodeBrain native pretraining path runs end-to-end on a tiny TUEG 19-common subset:
  pretrain_EEGSSM.py -> EEGSSM_Trainer -> SSSM(if_codebook=True) + frozen released TFDual tokenizer,
  masked dual-token cross-entropy. Then save a checkpoint and reload it byte/param-exact.

FAITHFULNESS / what is native vs. bounded (documented for the record):
  - Tokenizer: native `create_model('tfdual_vq', pretrained=True, ...)` (Models.modeling_tokenizer). Native
    unwraps {'model': ...}. Frozen (.eval(), no_grad, no EMA update).
  - Model: native Models.SSSM.SSSM(if_codebook=True), constructed with pretrain_EEGSSM.py defaults.
  - Trainer: native Pretrain.Trainer.EEGSSM_Trainer is CONSTRUCTED (exercises native model.to/vqnsp.to/
    channel_list/criterion/vqloss/AdamW/CosineAnnealingLR). Its __init__ profiling calls (torchinfo.summary +
    ptflops.get_model_complexity_info) are monkeypatched to no-ops BEFORE construction: they profile with B=1,
    which is incompatible with SSSM's if_codebook mask path (x.squeeze() drops the batch dim at B==1 and mask is
    None) -- a cosmetic native-code quirk orthogonal to training. This is a profiling bypass, NOT a
    reimplementation of masking or the tokenizer.
  - Mask: native Utils.util.generate_mask -> (B,19,30) 3-D mask (NOT flattened). The (B,570) flatten is done
    trainer-internal (rearrange 'b s c -> b (s c)') AFTER the SSSM forward, exactly as Trainer.train (~line 282).
  - Loop body: identical statements to EEGSSM_Trainer.train() masked branch (Trainer.py ~271-303), operating on
    the native trainer's own objects, bounded to --max-steps with per-step CE logging.

NORMALIZATION: CodeBrain expects uV-scale input (native does x/100). The processed TUEG corpus (4704743c) is
stored in VOLTS (measured window std ~1.5e-5 V = 15 uV). So for CodeBrain we feed uV = raw_volts * 1e6 and the
loop divides by 100 (native). We DO NOT z-score (the shared loader `windows_for` z-scores for CBraMod; z-score
would make the frozen tokenizer OOD). The shared loader is NOT modified; uV extraction is replicated here.
"""
import argparse, json, os, sys, time, traceback
import numpy as np
import torch

SMOKE_DIR = os.path.dirname(os.path.abspath(__file__))
CODEBRAIN = "/home/infres/yinwang/CodeBrain"
LOADER_DIR = "/home/infres/yinwang/CMI_AAAI_s2p/s2p/scripts"
TOKENIZER_CK = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth"
for p in (CODEBRAIN, LOADER_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()


def load_uv_windows(n_subjects, hours_budget, seed, target_windows, max_win_per_rec, fh):
    """Build in-memory uV windows (raw_volts * 1e6, NO z-score) from the TUEG 19-common subset.
    Reason-code every skipped/short recording; fail loud on unexpected shape."""
    import tueg_subject_loader as Lz
    sub = Lz.build_subset(n_subjects=n_subjects, hours_budget=hours_budget, condition="fixed_hours", seed=seed)
    log(f"build_subset manifest: {json.dumps(sub['manifest'])}", fh)
    rows = sub["train"]
    m = Lz._meta()
    WLEN, N_PATCH, PATCH = Lz.WLEN, Lz.N_PATCH, Lz.PATCH
    xs = []
    used_rows = 0; skipped = {}
    for r in rows:
        if sum(x.shape[0] for x in xs) >= target_windows:
            break
        try:
            chn = json.loads(r["channels"]); idx = [chn.index(c) for c in Lz.COMMON19]
        except Exception as e:
            skipped[f"row{r['recording_id']}:chan_index:{e!r}"] = 1; continue
        a = np.load(f"{Lz.TUEG}/{r['filepath']}", mmap_mode="r")
        T = a.shape[0]; nwin = T // WLEN
        if r.get("take_windows"):
            nwin = min(nwin, int(r["take_windows"]))
        nwin = min(nwin, max_win_per_rec)
        if nwin == 0:
            skipped[f"row{r['recording_id']}:zero_windows(T={T})"] = 1; continue
        x = np.asarray(a[: nwin * WLEN, idx], dtype=np.float32)
        x = x.reshape(nwin, N_PATCH, PATCH, 19).transpose(0, 3, 1, 2)   # (nwin,19,30,200)
        x = x * 1e6                                                     # VOLTS -> uV  (NO z-score)
        if x.shape[1:] != (19, N_PATCH, PATCH):
            raise RuntimeError(f"bad window shape {x.shape} for rec {r['recording_id']}")
        xs.append(x.astype(np.float32)); used_rows += 1
    if not xs:
        raise RuntimeError("no windows loaded from subset")
    X = np.concatenate(xs, axis=0)[:target_windows]
    if skipped:
        log(f"skipped rows (reason-coded): {json.dumps(skipped)}", fh)
    log(f"loaded uV windows: X.shape={list(X.shape)} from {used_rows} recordings; "
        f"uV std={float(X.std()):.4f} absmax={float(np.abs(X).max()):.4f} "
        f"(post /100 -> std={float((X/100).std()):.5f})", fh)
    return torch.from_numpy(X)


def build_sssm():
    from Models.SSSM import SSSM
    return SSSM(in_channels=200, res_channels=200, skip_channels=200, out_channels=200, num_res_layers=8,
                diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
                s4_lmax=570, s4_d_state=64, s4_dropout=0.1, s4_bidirectional=True, s4_layernorm=True,
                codebook_size_t=4096, codebook_size_f=4096, if_codebook=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=32)
    ap.add_argument("--hours-budget", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)      # >=2 required (SSSM squeezes batch dim at B==1)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--target-windows", type=int, default=2048)
    ap.add_argument("--max-win-per-rec", type=int, default=48)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(SMOKE_DIR, exist_ok=True)
    log_path = os.path.join(SMOKE_DIR, "codebrain_native_smoke.log")
    json_path = os.path.join(SMOKE_DIR, "codebrain_native_smoke_go_nogo.json")
    ckpt_path = os.path.join(SMOKE_DIR, "codebrain_smoke_ckpt.pth")
    fh = open(log_path, "a")

    gonogo = dict(ran_end_to_end=False, steps_completed=0, dual_token_ce_decreasing=None,
                  checkpoint_saved=False, checkpoint_reload_exact=None, blocker=None,
                  device_config=None, mask_shape_used=None,
                  ce_first20_mean=None, ce_last20_mean=None, ce_step0=None, ce_final=None,
                  native_trainer_constructed=False, notes="infra smoke only; NO science claim; NO P1 inclusion")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in job (SSSM hardcodes .cuda(); GPU required)")
        setup_seed = args.seed
        torch.manual_seed(setup_seed); np.random.seed(setup_seed)
        device = torch.device("cuda:0")   # SLURM CUDA_VISIBLE_DEVICES remaps physical GPU -> cuda:0
        gonogo["device_config"] = dict(parallel=False, cuda_index_in_job=0,
                                       CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       device="cuda:0", gpu_name=torch.cuda.get_device_name(0))
        log(f"device_config: {json.dumps(gonogo['device_config'])}", fh)

        # ---- data (uV, no z-score) ----
        X = load_uv_windows(args.n_subjects, args.hours_budget, args.seed,
                            args.target_windows, args.max_win_per_rec, fh)
        ds = torch.utils.data.TensorDataset(X)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         drop_last=True, num_workers=0)
        log(f"dataloader: n_windows={len(ds)} batch_size={args.batch_size} n_batches/epoch={len(dl)}", fh)

        # ---- native tokenizer (frozen) ----
        import Models.modeling_tokenizer  # registers tfdual_vq
        from timm.models import create_model
        tok = create_model("tfdual_vq", pretrained=True, pretrained_weight=TOKENIZER_CK, as_tokenzer=True,
                           n_code_t=4096, n_code_f=4096, code_dim=32).eval().to(device)
        for p in tok.parameters():
            p.requires_grad_(False)
        log(f"tokenizer loaded (frozen): n_params={sum(p.numel() for p in tok.parameters())}", fh)

        # ---- native model ----
        model = build_sssm().to(device)
        log(f"SSSM built: n_params={sum(p.numel() for p in model.parameters())}", fh)

        # ---- native EEGSSM_Trainer construction (profiling monkeypatched to no-op) ----
        import Pretrain.Trainer as TrainerMod
        TrainerMod.summary = lambda *a, **k: None                       # torchinfo.summary (B=1, cosmetic)
        TrainerMod.get_model_complexity_info = lambda *a, **k: ("NA", "NA")  # ptflops (B=1, cosmetic)
        from types import SimpleNamespace
        params = SimpleNamespace(cuda=0, parallel=False, lr_scheduler="CosineAnnealingLR", lr=1e-4,
                                 weight_decay=5e-3, epochs=1, need_mask=True, mask_ratio=args.mask_ratio,
                                 clip_value=5.0, model_dir=SMOKE_DIR)
        trainer = TrainerMod.EEGSSM_Trainer(params, dl, model, tok)     # native init path
        gonogo["native_trainer_constructed"] = True
        log("native EEGSSM_Trainer constructed (profiling no-op'd; model/vqnsp/optimizer/scheduler/vqloss native)", fh)

        # ---- bounded native loop (Trainer.train masked branch, verbatim body, capped + per-step logging) ----
        from einops import rearrange
        from Utils.util import generate_mask
        m, vqnsp = trainer.model, trainer.vqnsp
        opt, sched, vqloss = trainer.optimizer, trainer.optimizer_scheduler, trainer.vqloss
        chan = trainer.channel_list
        m.train()
        ce_hist = []
        step = 0
        done = False
        while not done:
            for (xb,) in dl:
                opt.zero_grad()
                x = xb.to(device) / 100                                 # native /100 (uV -> model scale)
                bz, ch_num, patch_num, patch_size = x.shape
                mask = generate_mask(bz, ch_num, patch_num, mask_ratio=params.mask_ratio, device=device)  # (B,19,30)
                if gonogo["mask_shape_used"] is None:
                    gonogo["mask_shape_used"] = list(mask.shape)
                y_t, y_f = m(x, mask=mask)
                with torch.no_grad():
                    input_t_ids, input_f_ids = vqnsp.get_codebook_indices(x, chan)
                    mflat = rearrange(mask, "b s c -> b (s c)")
                    codes_t, codes_f = input_t_ids[mflat == 1], input_f_ids[mflat == 1]
                loss = vqloss(y_t, codes_t) + vqloss(y_f, codes_f)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), params.clip_value)
                opt.step(); sched.step()
                lv = float(loss.detach().cpu())
                ce_hist.append(lv)
                if step < 3 or step % 20 == 0:
                    log(f"step {step}: dual_token_CE={lv:.4f} (n_masked_t={codes_t.numel()})", fh)
                step += 1
                if step >= args.max_steps:
                    done = True; break

        gonogo["steps_completed"] = step
        gonogo["ran_end_to_end"] = step >= args.max_steps
        gonogo["ce_step0"] = ce_hist[0]; gonogo["ce_final"] = ce_hist[-1]
        f20 = float(np.mean(ce_hist[:20])); l20 = float(np.mean(ce_hist[-20:]))
        gonogo["ce_first20_mean"] = f20; gonogo["ce_last20_mean"] = l20
        gonogo["dual_token_ce_decreasing"] = bool(l20 < f20)
        log(f"loop done: steps={step} CE first20={f20:.4f} last20={l20:.4f} "
            f"decreasing={gonogo['dual_token_ce_decreasing']}", fh)

        # ---- checkpoint save (native style: state_dict of the model) ----
        torch.save(m.state_dict(), ckpt_path)
        gonogo["checkpoint_saved"] = os.path.exists(ckpt_path)
        log(f"checkpoint saved: {ckpt_path} ({os.path.getsize(ckpt_path)} bytes)", fh)

        # ---- checkpoint reload + param-exact verification ----
        sd_ref = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m2 = build_sssm().to(device)
        loadres = m2.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=True)
        assert len(loadres.missing_keys) == 0 and len(loadres.unexpected_keys) == 0, f"load mismatch {loadres}"
        sd2 = m2.state_dict()
        all_exact = True; nkeys = 0
        for k, v in sd_ref.items():
            nkeys += 1
            if not torch.equal(v, sd2[k].detach().cpu()):
                all_exact = False
                log(f"  reload MISMATCH on key {k}", fh)
        gonogo["checkpoint_reload_exact"] = bool(all_exact)
        log(f"checkpoint reload: strict-load OK, {nkeys} tensors compared, param-exact={all_exact}", fh)

    except Exception as e:
        gonogo["blocker"] = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log("BLOCKER:\n" + tb, fh)
    finally:
        with open(json_path, "w") as jf:
            json.dump(gonogo, jf, indent=2)
        log(f"go/no-go written: {json_path}", fh)
        verdict = "PASS" if (gonogo["ran_end_to_end"] and gonogo["checkpoint_saved"]
                             and gonogo["checkpoint_reload_exact"]) else "BLOCKER"
        log(f"VERDICT: {verdict}", fh)
        fh.close()
        print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
