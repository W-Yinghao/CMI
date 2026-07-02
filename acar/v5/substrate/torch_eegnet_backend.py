"""ACAR V5 Stage-1B REAL numeric backend (torch imported LAZILY inside methods; nothing heavy at module load). Implements the two
seams:
  * fit(train, val, training_config): trains the pinned EEGNet deterministically on the FIT-train windows, early-stops on FIT-val,
    fits the class-conditional-Gaussian source state on the frozen ENCODER features, and returns the 4 model artifacts as
    DETERMINISTIC pickle-free bytes (encoder_state_dict / encoder_checkpoint_file / source_state_artifact / source_state_file).
  * embed_from_artifacts(windows_by_subject, frozen, training_config): rebuilds the EEGNet, loads the FROZEN encoder state from the
    handle's checkpoint file (no shared in-memory trainer state), and returns per-window ENCODER features for EXACTLY every requested
    subject (rows == that subject's n_windows).

Determinism: set_deterministic pins torch determinism + seed + single-thread; training shuffles with a seeded numpy RandomState; the
serialization is byte-canonical (no pickle, no zip). NO real DEV read here — driven by already-read SubjectWindows.
"""
from __future__ import annotations
import hashlib
import json
from acar.v5.substrate import eegnet_architecture as EA
from acar.v5.substrate import source_state as SS

N_CHANS, N_TIMES, N_CLASSES = 19, 512, 2


class TorchEegnetBackendError(RuntimeError):
    pass


def _make_xy(records, torch, np):
    """(subject_key, SubjectWindows, label) triples → (X: (Nwin,1,19,512) float32 tensor, y: (Nwin,) long tensor)."""
    xs, ys = [], []
    for _sk, sw, label in records:
        w = np.asarray(sw.windows, dtype=np.float32)
        if w.ndim != 3 or w.shape[1:] != (N_CHANS, N_TIMES):
            raise TorchEegnetBackendError(f"windows must be (n,{N_CHANS},{N_TIMES}), got {w.shape}")
        xs.append(w)
        ys.append(np.full((w.shape[0],), int(label), dtype=np.int64))
    X = np.concatenate(xs, axis=0)[:, None, :, :]             # (Nwin, 1, 19, 512)
    y = np.concatenate(ys, axis=0)
    return torch.from_numpy(X), torch.from_numpy(y)


class TorchEegnetBackend:
    def set_deterministic(self, seed):
        import torch  # lazy — never imported at module load
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(seed))
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass
        torch.set_num_threads(1)
        self._seed = int(seed)

    def fit(self, train, val, training_config):
        import copy
        import torch
        import torch.nn as nn
        import numpy as np
        if not hasattr(self, "_seed"):                        # determinism contract: seed must be pinned before fit
            raise TorchEegnetBackendError("set_deterministic(seed) must be called before fit()")
        seed = int(self._seed)
        model = EA.build_eegnet(N_CHANS, N_TIMES, N_CLASSES)
        opt = torch.optim.Adam(model.parameters(), lr=float(training_config["lr"]),
                               weight_decay=float(training_config["weight_decay"]))
        loss_fn = nn.CrossEntropyLoss()
        Xtr, ytr = _make_xy(train, torch, np)
        Xva, yva = _make_xy(val, torch, np)
        batch = int(training_config["batch_size"])
        rng = np.random.RandomState(seed)
        best_state, best_val, bad = copy.deepcopy(model.state_dict()), float("inf"), 0
        patience = int(training_config["early_stopping_patience"])
        for _epoch in range(int(training_config["max_epochs"])):
            model.train()
            idx = rng.permutation(Xtr.shape[0])
            for start in range(0, len(idx), batch):
                b = torch.from_numpy(idx[start:start + batch])
                opt.zero_grad()
                loss = loss_fn(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(Xva), yva).item())
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            feats = model.encode(Xtr).cpu().numpy()
        state = SS.fit_source_state(feats, ytr.cpu().numpy())
        ss_file = SS.serialize_source_state(state)
        full_ckpt = EA.canonical_state_bytes(model.state_dict())
        enc_sd = EA.canonical_state_bytes({k: v for k, v in model.state_dict().items() if not k.startswith("classifier.")})
        ss_artifact = json.dumps({"schema": "acar_v5_source_state_v1", "source_state_sha256": hashlib.sha256(ss_file).hexdigest(),
                                  "embedding_dim": int(model.embedding_dim), "n_classes": N_CLASSES},
                                 sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {"encoder_state_dict": enc_sd, "encoder_checkpoint_file": full_ckpt,
                "source_state_artifact": ss_artifact, "source_state_file": ss_file}

    def embed_from_artifacts(self, windows_by_subject, frozen, training_config):
        import torch
        import numpy as np
        model = EA.build_eegnet(N_CHANS, N_TIMES, N_CLASSES)
        with open(frozen.encoder_checkpoint_file_path, "rb") as f:
            blob = f.read()
        EA.load_state_into(model, blob)                       # load FROZEN encoder state from the handle (no shared trainer state)
        model.eval()
        out = {}
        for sk, sw in windows_by_subject.items():
            w = np.asarray(sw.windows, dtype=np.float32)
            if w.ndim != 3 or w.shape[1:] != (N_CHANS, N_TIMES):
                raise TorchEegnetBackendError(f"{sk}: windows must be (n,{N_CHANS},{N_TIMES}), got {w.shape}")
            X = torch.from_numpy(w)[:, None, :, :]
            with torch.no_grad():
                out[sk] = model.encode(X).cpu().numpy().astype("float32", copy=False)
        return out
