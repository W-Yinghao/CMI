"""CMI-Trace Version 1 — Fixed-P Train-Through-Erasure (TTE).

The post-hoc ladder found that deleting a source-estimated subject subspace from an ALREADY-TRAINED
representation does not help — but the task head there was trained on the FULL Z and only saw the erased
representation at audit time. TTE tests the methodologically different question:

    if the model is forced to predict from the ERASED representation DURING training, can the encoder route
    the task information OUT of the deleted subspace and into the kept branch?

Protocol (Version 1, fixed projector):
  1. ERM warm-up (reuse the standard trainer) -> encoder f_theta + head.
  2. Estimate a FIXED projector P_0 SOURCE-ONLY (exact-head-null / label-conditional subject / random / none).
  3. Insert (I - P_0) before the head: logits = head( (I-P_0) graph_z ). Freeze the LOWER encoder
     (enc/conv/adj); fine-tune only the TOP block (readout proj) + a FRESH head, through the erasure.
  4. Score target (eval-only). Compare arms {full(no erasure), exact_head_null, subject, random}.

Success is NOT "target accuracy up". Pre-frozen criteria: kept-branch CMI down, exact-head reliance not up,
source task retained (>= -0.02), and the informed projector's target effect beats a same-rank random
projector. Operates on the DGCNN forward-graph adapter. Requires torch; training runs on GPU.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _lower_top_params(adapter):
    """Split DGCNNForwardGraphAdapter params into LOWER (enc/conv/adj) vs TOP (proj + head)."""
    net = adapter.net
    lower, top = [], []
    for name, p in adapter.named_parameters():
        if any(k in name for k in (".enc", ".conv", ".adj")):
            lower.append(p)
        else:                                              # proj + head (readout + classifier)
            top.append(p)
    return lower, top


def _erased_logits(adapter, x, Pt):
    """logits = head( (I - Pt) graph_z ); graph_z = readout(node_z). Pt is a [d,d] torch projector buffer."""
    node_z = adapter._node_z(x)                            # lower
    graph_z = adapter._readout(node_z)                    # top proj (trainable)
    d = graph_z.shape[1]
    graph_z_e = graph_z @ (torch.eye(d, device=graph_z.device) - Pt).t()
    return adapter.net.head(graph_z_e), graph_z, graph_z_e


def train_through_erasure(adapter, P0, Xtr, ytr, *, freeze_lower=True, reinit_head=True,
                          epochs=60, bs=64, lr=1e-3, weight_decay=1e-4, device="cpu", seed=0):
    """Fine-tune the TOP block + a fresh head so the model predicts from (I-P0) graph_z. P0 numpy [d,d].
    Returns the adapter with a fixed erasure buffer `tte_P` and helpers. Lower encoder frozen if requested."""
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    Pt = torch.tensor(np.asarray(P0, float), dtype=torch.float32, device=device)
    adapter.to(device)
    lower, top = _lower_top_params(adapter)
    if freeze_lower:
        for p in lower:
            p.requires_grad_(False)
    if reinit_head:
        adapter.net.head.reset_parameters()
    train_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TensorDataset(torch.tensor(np.asarray(Xtr), dtype=torch.float32),
                       torch.tensor(np.asarray(ytr), dtype=torch.long))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(ds) % bs == 1))
    adapter.train()
    for _ in range(int(epochs)):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, _ = _erased_logits(adapter, xb, Pt)
            opt.zero_grad(); F.cross_entropy(logits, yb).backward(); opt.step()
        sched.step()
    adapter.tte_P = Pt
    return adapter


@torch.no_grad()
def predict_erased(adapter, X, P0=None, device="cpu", bs=512):
    """Predict with the erasure applied before the head (uses adapter.tte_P if P0 is None)."""
    adapter.eval()
    Pt = adapter.tte_P if P0 is None else torch.tensor(np.asarray(P0, float), dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(np.asarray(X[i:i + bs]), dtype=torch.float32).to(device)
        logits, _, _ = _erased_logits(adapter, xb, Pt)
        out.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def kept_graph_z(adapter, X, P0, device="cpu", bs=512):
    """The deployed kept-branch representation (I-P0) graph_z after TTE fine-tuning (for CMI/reliance audit)."""
    adapter.eval()
    Pt = torch.tensor(np.asarray(P0, float), dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(np.asarray(X[i:i + bs]), dtype=torch.float32).to(device)
        _, _, gz_e = _erased_logits(adapter, xb, Pt)
        out.append(gz_e.cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------- source-only projectors (in graph_z space)
def subject_projector(graph_z, y, d, k):
    """Label-conditional subject subspace projector P = S^T S (rank<=k) from source graph_z (the 'TOS-style'
    informed projector; orthogonal)."""
    from cmi.eval.leakage_removal import fit_leakage_subspace
    _, dirs = fit_leakage_subspace(graph_z, y, d, k, conditioning="label_conditional")
    return dirs.T @ dirs


def exact_head_null_projector_gz(graph_z, y, d, W, k):
    """Exact-head-null projector in graph_z space (range subseteq ker(W_c))."""
    from tos_cmi.eeg.erasure_oracle import exact_head_null_projector
    P, rank, _ = exact_head_null_projector(graph_z, y, d, W, k)
    return P, rank


def random_projector(d, k, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q @ Q.T
