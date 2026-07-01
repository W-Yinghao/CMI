"""Backbone wrapper: FAITHFUL braindecode encoder for the task + a hooked penultimate
feature Z used only by the LPC-CMI domain posterior.

Key design (so ERM == standard braindecode model, a credible baseline): we do NOT
replace the model's classifier. The native model produces the task logits; we register
a forward-pre-hook on `final_layer` to grab its input (the penultimate feature map),
global-pool it to a vector Z, and expose (logits, Z). The regularizer shapes the encoder
through Z; the task head is the model's own. EEGConformer uses native return_features.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

# braindecode (and its moabb dependency) is imported LAZILY, only when a braindecode task backbone is
# actually built. This keeps non-braindecode backbones (DGCNNGraph / GraphCMI / DGCNN / LogCov / TSMNet)
# importable even if the installed braindecode/moabb versions are incompatible in the current env.


def _braindecode_models():
    from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net, EEGConformer
    return {"EEGNet": EEGNetv4, "ShallowConvNet": ShallowFBCSPNet, "Deep4Net": Deep4Net}, EEGConformer


class HookedBackbone(nn.Module):
    def __init__(self, name, n_chans, n_times, n_classes):
        super().__init__()
        self.name = name
        _CONV, EEGConformer = _braindecode_models()
        self._feat = None
        if name != "EEGConformer" and name not in _CONV:
            raise ValueError(f"unknown backbone {name}")
        # Build + probe z_dim. Some backbones only accept certain n_times (verified:
        # EEGConformer {1000,1001}; Deep4Net needs n_times>=~450 and fails at CONSTRUCTION
        # with final_conv_length='auto'). Wrap both construction and the dummy forward, and
        # re-raise the cryptic shape error with guidance — no fragile hard-coded thresholds.
        # The dummy forward runs in eval() so BatchNorm running stats stay untouched.
        try:
            if name == "EEGConformer":
                self.model = EEGConformer(n_chans=n_chans, n_outputs=n_classes,
                                          n_times=n_times, return_features=True)
                self._mode = "conformer"
            else:
                kw = dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
                if name in ("ShallowConvNet", "Deep4Net"):
                    kw["final_conv_length"] = "auto"  # pool time -> (B, n_classes) for any n_times
                self.model = _CONV[name](**kw)
                self.model.final_layer.register_forward_pre_hook(self._capture)
                self._mode = "conv"
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    _, z = self.forward(torch.zeros(2, n_chans, n_times))
            finally:
                self.train(was_training)
        except Exception as e:
            hint = ("EEGConformer needs n_times==1000 (250Hz x 4s: --tmin 0 --tmax 4 --resample 250)."
                    if name == "EEGConformer" else
                    "Deep4Net needs a longer window (n_times>=~450); use --resample 250."
                    if name == "Deep4Net" else "try a longer window / higher --resample.")
            raise ValueError(f"{name} cannot build at n_times={n_times} ({type(e).__name__}). {hint}") from e
        self.z_dim = z.shape[1]

    def _capture(self, module, inp):
        f = inp[0]                                   # penultimate feature map
        self._feat = f.mean(dim=tuple(range(2, f.dim()))) if f.dim() > 2 else f

    def forward(self, x):
        if self._mode == "conformer":
            logits, z = self.model(x)                # native (logits, 32-d feature)
        else:
            logits = self.model(x)                   # native logits; hook fills self._feat
            z = self._feat
        return logits, z


class TSMNetBackbone(nn.Module):
    """Wraps the official rkobler/TSMNet (repos/TSMNet) as a (logits, Z) backbone.
    Uses bnorm='spdbn' (non-domain-specific SPD BatchNorm) -> a calibration-free DG SPDNet
    (NOT the SPDDSMBN/UDA variant, which would use target data). Z = LogEig tangent latent.
    NOTE: its BiMap weights live on the Stiefel manifold -> needs a Riemannian optimizer
    (the trainer switches to geoopt RiemannianAdam when manifold params are present)."""
    def __init__(self, n_chans, n_times, n_classes, device="cpu", temporal_filters=4):
        super().__init__()
        import sys
        repo = Path(__file__).resolve().parents[2] / "repos" / "TSMNet"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from spdnets.models import TSMNet
        dev = torch.device(device)
        self.model = TSMNet(temporal_filters=temporal_filters, nclasses=n_classes,
                            nchannels=n_chans, nsamples=n_times, bnorm="spdbn", device=dev)
        # The repo hardcodes the SPD layers to CPU (line 23). Override so they run on `device`;
        # all SPD modules use self.spd_device_ for both creation and the per-forward cast, and
        # .to(dev) moves the already-created CPU SPD params to the GPU. (torch 2.8 supports CUDA
        # double-precision eigh, so the BiMap/ReEig/LogEig/SPD-BN run on GPU.)
        self.model.spd_device_ = dev
        self.model.to(dev)
        with torch.no_grad():
            out = self.model(torch.zeros(2, n_chans, n_times, device=dev),
                             torch.zeros(2, dtype=torch.long, device=dev), return_latent=True)
        self.z_dim = out[1].shape[1]

    def forward(self, x):
        d = torch.zeros(x.size(0), dtype=torch.long, device=x.device)   # unused for spdbn
        out = self.model(x, d, return_latent=True)
        return out[0], out[1].float()                 # (logits, tangent latent Z)


class MLPBackbone(nn.Module):
    """Small MLP on a flat feature vector (e.g. LogCov tangent vectors) -> (logits, Z).
    Used for the 'LogCov' carrier; the geometric features are precomputed in the runner."""
    def __init__(self, n_chans, n_times, n_classes, z_dim=64, hidden=256):
        super().__init__()
        din = n_chans * n_times
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, z_dim), nn.ReLU())
        self.task_head = nn.Linear(z_dim, n_classes)
        self.z_dim = z_dim

    def forward(self, x):
        z = self.net(x.flatten(1))
        return self.task_head(z), z


def build_backbone(name, n_chans, n_times, n_classes, device="cpu", **_):
    if name == "LogCov":
        return MLPBackbone(n_chans, n_times, n_classes).to(device)
    if name == "TSMNet":
        return TSMNetBackbone(n_chans, n_times, n_classes, device=device).to(device)
    if name in ("GraphCMI", "DGCNN", "RGNN"):
        from cmi.models import gnn
        cls = {"GraphCMI": gnn.GraphCMINet, "DGCNN": gnn.DGCNNBackbone, "RGNN": gnn.RGNNBackbone}[name]
        return cls(n_chans, n_times, n_classes).to(device)
    if name == "DGCNNGraph":
        # Graph-DualCMI: the task-capable static-adjacency DGCNN adapter exposing
        # forward_graph(x) -> (logits, graph_z, node_z, edge_logits=None). Distinct name from "DGCNN"
        # (which maps to gnn.DGCNNBackbone and has NO forward_graph); this is the one graph/node CMI +
        # decoder-CMI methods (graphcmi / graphdualpc) can run on. edge_logits is None (static adjacency).
        from cmi.models.graph_task_backbones import DGCNNForwardGraphAdapter
        return DGCNNForwardGraphAdapter(n_chans, n_times, n_classes).to(device)
    if name == "FBLGGGraph":
        # CIGL_47 main line: FilterBank temporal + Local-Global electrode Graph + gated fusion.
        # forward_graph(x) -> (logits, graph_z, node_z, edge_logits=None, fused_z) — a 5-tuple with a
        # DISTINCT fused_z (the classifier input), so graphdualpc runs a genuine encoder/decoder head
        # split. ch_names (optional) enables name-aware electrode grouping; absent -> index partition.
        from cmi.models.fb_lgg_dualcmi import FBLGGDualCMIBackbone
        return FBLGGDualCMIBackbone(n_chans, n_times, n_classes, ch_names=_.get("ch_names"),
                                    groups=_.get("groups"), group_names=_.get("group_names"),
                                    grouping_scheme=_.get("grouping_scheme")).to(device)
    return HookedBackbone(name, n_chans, n_times, n_classes).to(device)
