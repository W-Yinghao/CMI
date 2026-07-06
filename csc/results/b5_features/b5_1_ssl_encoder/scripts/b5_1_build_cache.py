"""B5.1 SSL masked-reconstruction EEGNet encoder cache (development-only; feature-dependence test; NOT deployable).
The B5.0 red-team showed a FROZEN RANDOM encoder is a near-isometry NO-OP (reproduces SM16, no concept power), so it
cannot answer whether a LEARNED representation changes the fitted-null FAIL. B5.1 trains the SAME EEGNet trunk with a
LABEL-FREE self-supervised objective (masked time-span reconstruction) -> a LEARNED 192-dim embedding Z_ssl, then
freezes it and extracts the penultimate embedding exactly like B5.0.

LEAKAGE GUARANTEE (same rigor as B5.0, plus a training phase): the SSL loss uses ONLY the raw signal X. It never
sees any label -- not the real MI label, not a synthetic/injected label, not session, not subject. It is pure signal
reconstruction. y (real MI) is stored separately for the downstream injection bank ONLY; injection labels are
regenerated ON Z_ssl. The encoder weight hash is recorded AFTER training.
CAVEAT (recorded): SSL is transductive (trained on all windows it then embeds) -- benign for a LABEL-leakage-free
canary (same posture as B5.0's global standardization), but noted; a future inductive variant would split.
"""
import os, sys, json, hashlib, argparse, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc/csc/mininfo")
import build_lee2019_b3_cache as B
import b5_0_build_cache as B50            # reuse raw_windows_for_run + EEGNetEncoder (identical trunk)
import torch

SEED = 20260707                            # distinct from B5.0's 20260706
MASK_RATIO = 0.5                           # fraction of time columns masked (whole-electrode-column, standard EEG MAE)
EPOCHS = 60
BATCH = 256
LR = 1e-3


class Decoder(torch.nn.Module):
    """Mirror of EEGNetEncoder: 192-dim embedding -> reconstructed [N,1,16,384]. Verified dimensionally at build."""
    def __init__(self, F2=16, t_emb=12, C=16, T=384):
        super().__init__()
        self.F2, self.t_emb = F2, t_emb
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(F2, F2, (1, 8), stride=(1, 8)),   # time 12 -> 96
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(F2, F2, (1, 4), stride=(1, 4)),   # time 96 -> 384
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(F2, 1, (C, 1)))                   # spatial 1 -> C(16)
    def forward(self, z):                                             # z [N, F2*t_emb]
        h = z.view(z.shape[0], self.F2, 1, self.t_emb)
        return self.net(h)                                           # [N,1,C,T]


def make_mask(n, T, ratio, gen):
    """Boolean [n,T] mask: True = MASKED (zeroed in input; loss computed here). Whole time-columns."""
    k = int(round(ratio * T))
    m = torch.zeros(n, T, dtype=torch.bool)
    for i in range(n):
        idx = torch.randperm(T, generator=gen)[:k]
        m[i, idx] = True
    return m


def build(n_subjects, out_dir, device):
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    # ---- raw windows (identical extraction to B5.0) ----
    Xs, ys, subj, sess, tid, per_file, missing_ch = [], [], [], [], [], [], []
    for s in range(1, n_subjects + 1):
        for ss in B.SESSIONS:
            p = B.mat_path(s, ss)
            if not os.path.exists(p): continue
            try: X, y = B50.raw_windows_for_run(p)
            except RuntimeError as e:
                if "FAIL_CLOSED" in str(e): missing_ch.append([s, ss, str(e)]); continue
                raise
            Xs.append(X); ys.append(y); subj.append(np.full(len(y), s)); sess.append(np.full(len(y), ss)); tid.append(np.arange(len(y)))
            per_file.append(dict(file=p, size=os.path.getsize(p), n=int(len(y)),
                                 sha1_10=hashlib.sha1(open(p, 'rb').read(1 << 20)).hexdigest()[:10]))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    subject_id = np.concatenate(subj); session_id = np.concatenate(sess); trial_id = np.concatenate(tid)
    C, T = X.shape[1], X.shape[2]
    print(f"raw windows: X {X.shape} (N,C,T) y {y.shape}  ({time.time()-t0:.0f}s)", flush=True)

    torch.use_deterministic_algorithms(True, warn_only=True); torch.manual_seed(SEED)
    enc = B50.EEGNetEncoder(C=C, T=T).to(device)
    emb_dim = enc(torch.zeros(1, 1, C, T, device=device)).shape[1]
    t_emb = emb_dim // 16
    dec = Decoder(F2=16, t_emb=t_emb, C=C, T=T).to(device)
    # dimensional self-check
    with torch.no_grad():
        chk = dec(enc(torch.zeros(2, 1, C, T, device=device)))
    assert chk.shape == (2, 1, C, T), f"decoder shape {chk.shape} != (2,1,{C},{T})"
    print(f"emb_dim={emb_dim} t_emb={t_emb}; decoder shape OK {tuple(chk.shape)}", flush=True)

    # ---- SSL training: masked time-span reconstruction (LABEL-FREE; uses only X) ----
    Xt = torch.from_numpy(X)                         # [N,C,T] float32 on CPU; moved per-batch
    N = len(Xt)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    gen = torch.Generator().manual_seed(SEED + 1)
    perm_gen = torch.Generator().manual_seed(SEED + 2)
    enc.train(); dec.train()
    loss_curve = []
    for ep in range(EPOCHS):
        order = torch.randperm(N, generator=perm_gen)
        tot, nb = 0.0, 0
        for i in range(0, N, BATCH):
            bidx = order[i:i + BATCH]
            xb = Xt[bidx].to(device)                 # [n,C,T]
            m = make_mask(len(bidx), T, MASK_RATIO, gen).to(device)   # [n,T] True=masked
            # zero masked time-columns per-sample (all electrodes at masked times):
            xmask = xb.clone()
            xmask[m.unsqueeze(1).expand(-1, C, -1)] = 0.0
            z = enc(xmask.unsqueeze(1))              # [n,emb]
            rec = dec(z).squeeze(1)                  # [n,C,T]
            mm = m.unsqueeze(1).expand(-1, C, -1)    # [n,C,T]
            loss = ((rec - xb) ** 2)[mm].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        loss_curve.append(tot / max(nb, 1))
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ssl epoch {ep:3d}  masked-MSE {loss_curve[-1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- freeze + extract penultimate embedding on the FULL (unmasked) windows ----
    enc.eval()
    weight_hash = hashlib.sha256(np.concatenate([p.detach().cpu().numpy().ravel() for p in enc.parameters()]).tobytes()).hexdigest()
    def fwd(Xin):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xin), 512):
                xb = torch.from_numpy(Xin[i:i + 512]).unsqueeze(1).to(device)
                out.append(enc(xb).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    Z1 = fwd(X); Z2 = fwd(X[:1024])
    repro = float(np.abs(Z1[:1024] - Z2).max())
    mu = Z1.mean(0); sd = Z1.std(0) + 1e-8; Zssl = ((Z1 - mu) / sd).astype(np.float64)
    rank = int(np.linalg.matrix_rank(Zssl - Zssl.mean(0)))

    np.savez_compressed(os.path.join(out_dir, "LEE2019_B5_1.npz"),
        Z=Zssl, y=y, subject_id=subject_id, session_id=session_id, trial_id=trial_id,
        channel_names=np.array(B.MONTAGE), montage_name="SM16_no_FCz", feature_name="B5_1_ssl_eegnet")
    cache_sha = hashlib.sha256(open(os.path.join(out_dir, "LEE2019_B5_1.npz"), 'rb').read()).hexdigest()
    checks = dict(n_trials=int(len(y)), embedding_dim=emb_dim, embedding_rank=rank,
        repro_max_abs_diff=repro, repro_ok=bool(repro < 1e-6),
        nan=int(np.isnan(Zssl).sum()), inf=int(np.isinf(Zssl).sum()),
        emb_std_min=float(Zssl.std(0).min()), emb_std_median=float(np.median(Zssl.std(0))),
        subjects=int(len(np.unique(subject_id))), missing_channels=missing_ch,
        ssl_final_masked_mse=loss_curve[-1], ssl_initial_masked_mse=loss_curve[0], ssl_loss_curve=loss_curve)
    json.dump(checks, open(os.path.join(out_dir, "b5_1_cache_checks.json"), "w"), indent=1, default=str)
    manifest = dict(diagnostic_only=True, not_deployable=True, feature_family="B5_1_ssl_eegnet",
        stage="B5_1_ssl_masked_reconstruction_encoder", montage_name="SM16_no_FCz", channel_names=B.MONTAGE,
        bandpass=list(B.BANDPASS), window=list(B.WINDOW), fs_resampled=B.FS_RESAMPLED, n_times=int(T),
        preprocessing="IDENTICAL to SM16/B5.0 (8-30Hz Butterworth filtfilt, 0.5-3.5s, resample 384); differs from B5.0 ONLY in that the SAME EEGNet trunk is SSL-TRAINED (label-free) instead of random-init",
        encoder=dict(arch="EEGNet(F1=8,D=2,F2=16,kern=64,pool1=4,pool2=8), penultimate flatten, no classifier",
            random_init=False, trained=True, ssl_objective="masked_time_column_reconstruction",
            mask_ratio=MASK_RATIO, epochs=EPOCHS, batch=BATCH, lr=LR, optimizer="Adam",
            eval_mode=True, seed=SEED, weight_sha256=weight_hash, torch=torch.__version__, embedding_dim=emb_dim,
            ssl_final_masked_mse=loss_curve[-1]),
        normalization="per-dim z-score (label-free)", norm_mean_sha256=hashlib.sha256(mu.tobytes()).hexdigest()[:16],
        embedding_shape=[int(len(y)), emb_dim], cache_sha256=cache_sha,
        label_exposure=False, synthetic_label_exposure=False, real_MI_label_exposure=False,
        ssl_uses_only_signal=True, transductive_ssl=True,
        note="SSL objective is masked signal reconstruction using ONLY X; no MI/synthetic/session/subject label ever enters training or the embedding. y stored for the injection bank ONLY; injection labels regenerated ON Z_ssl. Transductive (trained on all embedded windows) -- recorded caveat.",
        source_files=per_file, build_seconds=round(time.time() - t0, 1))
    json.dump(manifest, open(os.path.join(out_dir, "b5_1_feature_manifest.json"), "w"), indent=2, default=str)
    print(f"EMB dim={emb_dim} rank={rank} repro_max={repro:.2e} repro_ok={repro<1e-6} nan={checks['nan']} inf={checks['inf']} std_med={checks['emb_std_median']:.3f}")
    print(f"ssl masked-MSE {loss_curve[0]:.5f} -> {loss_curve[-1]:.5f}  cache_sha256 {cache_sha[:16]} weight_sha256 {weight_hash[:16]}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=54)
    ap.add_argument("--out", default="/home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[b5.1] device={dev} torch={torch.__version__} cuda_avail={torch.cuda.is_available()} seed={SEED}", flush=True)
    build(a.subjects, a.out, dev)
    print("B5_1_BUILD_OK")
