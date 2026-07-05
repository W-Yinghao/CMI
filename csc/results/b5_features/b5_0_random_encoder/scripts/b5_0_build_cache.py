"""B5.0 frozen random-init EEGNet-style encoder cache (development-only; feature-robustness; NOT deployable;
NOT a rescue of the SM16 v2 FAIL). Reuses the EXACT SM16 preprocessing (8-30Hz Butterworth filtfilt, 0.5-3.5s,
resample 384, SM16_no_FCz 16ch) but keeps the raw [16x384] window; passes it through a FROZEN random-init
EEGNet (fixed seed, eval, dropout off) -> per-trial penultimate embedding Z_deep. The encoder NEVER sees any
label (real MI or synthetic). y (real MI) is stored separately, used ONLY by the injection bank (same as SM16).
Injection labels are regenerated ON Z_deep, not transplanted from SM16."""
import os, sys, json, hashlib, argparse, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample
sys.path.insert(0,"/home/infres/yinwang/CMI_AAAI_csc/csc/mininfo")
import build_lee2019_b3_cache as B    # parse_run, MONTAGE, BANDPASS, WINDOW, N_RESAMPLED, LABEL_MAP, mat_path, SESSIONS
import torch
SEED=20260706

def raw_windows_for_run(path):
    x, fs, chan, onsets, y_dec, parser = B.parse_run(path)
    missing=[c for c in B.MONTAGE if c not in chan]
    if missing: raise RuntimeError(f"FAIL_CLOSED montage channel(s) absent {missing} in {os.path.basename(path)}")
    idx=[chan.index(c) for c in B.MONTAGE]; xc=x[:,idx]
    sos=butter(4,[B.BANDPASS[0],B.BANDPASS[1]],btype="band",fs=fs,output="sos")
    xf=sosfiltfilt(sos,xc,axis=0); a,b=int(B.WINDOW[0]*fs),int(B.WINDOW[1]*fs)
    X,y=[],[]
    for t0,yd in zip(onsets,y_dec):
        seg=xf[t0+a:t0+b,:]
        if seg.shape[0] < (b-a)-2: continue
        seg=resample(seg,B.N_RESAMPLED,axis=0)          # [384,16] (same as SM16, pre log-var)
        X.append(seg.T.astype(np.float32)); y.append(B.LABEL_MAP.get(int(yd),-1))   # [16,384]
    return np.asarray(X,np.float32), np.asarray(y,np.int64)

class EEGNetEncoder(torch.nn.Module):
    """Standard EEGNet (Lawhern 2018) up to the penultimate flatten. No classifier head; leakage-free."""
    def __init__(self,C=16,T=384,F1=8,D=2,F2=16,kern=64,pool1=4,pool2=8):
        super().__init__()
        self.b1=torch.nn.Sequential(
            torch.nn.Conv2d(1,F1,(1,kern),padding=(0,kern//2),bias=False), torch.nn.BatchNorm2d(F1),
            torch.nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False), torch.nn.BatchNorm2d(F1*D),
            torch.nn.ELU(), torch.nn.AvgPool2d((1,pool1)))
        self.b2=torch.nn.Sequential(
            torch.nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),
            torch.nn.Conv2d(F1*D,F2,(1,1),bias=False), torch.nn.BatchNorm2d(F2),
            torch.nn.ELU(), torch.nn.AvgPool2d((1,pool2)))
    def forward(self,x):                                  # x [N,1,C,T]
        h=self.b2(self.b1(x)); return h.flatten(1)        # [N, F2 * (T//(pool1*pool2))]

def build(n_subjects, out_dir, device):
    t0=time.time(); os.makedirs(out_dir,exist_ok=True)
    Xs,ys,subj,sess,tid=[],[],[],[],[]; per_file=[]; missing_ch=[]
    for s in range(1,n_subjects+1):
        for ss in B.SESSIONS:
            p=B.mat_path(s,ss)
            if not os.path.exists(p): continue
            try: X,y=raw_windows_for_run(p)
            except RuntimeError as e:
                if "FAIL_CLOSED" in str(e): missing_ch.append([s,ss,str(e)]); continue
                raise
            Xs.append(X); ys.append(y); subj.append(np.full(len(y),s)); sess.append(np.full(len(y),ss)); tid.append(np.arange(len(y)))
            per_file.append(dict(file=p,size=os.path.getsize(p),n=int(len(y)),
                                 sha1_10=hashlib.sha1(open(p,'rb').read(1<<20)).hexdigest()[:10]))
    X=np.concatenate(Xs); y=np.concatenate(ys); subject_id=np.concatenate(subj); session_id=np.concatenate(sess); trial_id=np.concatenate(tid)
    print(f"raw windows: X {X.shape} (N,C,T) y {y.shape}  ({time.time()-t0:.0f}s)",flush=True)
    # frozen random encoder (fixed seed; leakage-free -- forward uses ONLY the signal, no y)
    torch.use_deterministic_algorithms(True); torch.manual_seed(SEED)
    enc=EEGNetEncoder(C=X.shape[1],T=X.shape[2]).to(device).eval()
    weight_hash=hashlib.sha256(np.concatenate([p.detach().cpu().numpy().ravel() for p in enc.parameters()]).tobytes()).hexdigest()
    def fwd(Xin):
        out=[]
        with torch.no_grad():
            for i in range(0,len(Xin),512):
                xb=torch.from_numpy(Xin[i:i+512]).unsqueeze(1).to(device)   # [n,1,C,T]
                out.append(enc(xb).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    Z1=fwd(X); Z2=fwd(X[:1024])                            # repro self-check on a slice
    repro=float(np.abs(Z1[:1024]-Z2).max())
    # standardize per-dim (LABEL-FREE) -> record stats
    mu=Z1.mean(0); sd=Z1.std(0)+1e-8; Zdeep=(Z1-mu)/sd
    Zdeep=Zdeep.astype(np.float64)
    emb_dim=Zdeep.shape[1]; rank=int(np.linalg.matrix_rank(Zdeep-Zdeep.mean(0)))
    # working cache (injection-ready; SM16 schema: Z,y,subject_id,session_id,trial_id)
    np.savez_compressed(os.path.join(out_dir,"LEE2019_B5_0.npz"),
        Z=Zdeep, y=y, subject_id=subject_id, session_id=session_id, trial_id=trial_id,
        channel_names=np.array(B.MONTAGE), montage_name="SM16_no_FCz", feature_name="B5_0_random_eegnet")
    cache_sha=hashlib.sha256(open(os.path.join(out_dir,"LEE2019_B5_0.npz"),'rb').read()).hexdigest()
    checks=dict(n_trials=int(len(y)), embedding_dim=emb_dim, embedding_rank=rank,
        repro_max_abs_diff=repro, repro_ok=bool(repro<1e-6),
        nan=int(np.isnan(Zdeep).sum()), inf=int(np.isinf(Zdeep).sum()),
        emb_std_min=float(Zdeep.std(0).min()), emb_std_median=float(np.median(Zdeep.std(0))),
        subjects=int(len(np.unique(subject_id))), missing_channels=missing_ch)
    json.dump(checks, open(os.path.join(out_dir,"b5_0_cache_checks.json"),"w"), indent=1, default=str)
    manifest=dict(diagnostic_only=True, not_deployable=True, not_a_rescue_of_SM16=True, feature_family="B5_0_random_eegnet",
        stage="B5_0_frozen_random_encoder", montage_name="SM16_no_FCz", channel_names=B.MONTAGE,
        bandpass=list(B.BANDPASS), window=list(B.WINDOW), fs_resampled=B.FS_RESAMPLED, n_times=int(X.shape[2]),
        preprocessing="IDENTICAL to SM16 (8-30Hz Butterworth filtfilt, 0.5-3.5s, resample 384); differs ONLY in representation (raw window -> frozen EEGNet embedding vs log-var)",
        encoder=dict(arch="EEGNet(F1=8,D=2,F2=16,kern=64,pool1=4,pool2=8), penultimate flatten, no classifier",
            random_init=True, trained=False, eval_mode=True, dropout_disabled=True, seed=SEED, weight_sha256=weight_hash,
            torch=torch.__version__, embedding_dim=emb_dim),
        normalization="per-dim z-score (label-free)", norm_mean_sha256=hashlib.sha256(mu.tobytes()).hexdigest()[:16],
        embedding_shape=[int(len(y)),emb_dim], cache_sha256=cache_sha,
        label_exposure=False, synthetic_label_exposure=False, real_MI_label_exposure=False,
        note="encoder forward used ONLY the raw signal; y (real MI) stored for the injection bank ONLY (same as SM16); injection labels regenerated ON Z_deep, NOT transplanted from SM16",
        source_files=per_file, build_seconds=round(time.time()-t0,1))
    json.dump(manifest, open(os.path.join(out_dir,"b5_0_feature_manifest.json"),"w"), indent=2, default=str)
    print(f"EMB dim={emb_dim} rank={rank} repro_max={repro:.2e} repro_ok={repro<1e-6} nan={checks['nan']} inf={checks['inf']} std_med={checks['emb_std_median']:.3f}")
    print(f"cache_sha256 {cache_sha[:16]} weight_sha256 {weight_hash[:16]}")
    return manifest

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--subjects",type=int,default=54); ap.add_argument("--out",default="/home/infres/yinwang/realeeg_feas/b5_features/b5_0_random_encoder")
    ap.add_argument("--device",default=None); a=ap.parse_args()
    dev=a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[b5.0] device={dev} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",flush=True)
    build(a.subjects,a.out,dev)
    print("B5_0_BUILD_OK")
