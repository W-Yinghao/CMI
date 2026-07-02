"""ACAR V5 Stage-1B EEGNet architecture (torch imported LAZILY inside functions; nothing heavy at module load). A compact,
deterministic EEGNet bound to the pinned training_config shape (19 channels × 512 samples, 2 classes). The ENCODER is the feature
extractor (everything before the linear classifier); `encode()` returns the per-window flattened features that become the routing
embedding. Also provides a CANONICAL, pickle-free state serialization (deterministic bytes) so the encoder can be hashed + reloaded
from the frozen artifacts.
"""
from __future__ import annotations
import json
import struct


def build_eegnet(n_chans, n_times, n_classes, *, F1=8, D=2, kern_len=64, drop=0.25):
    """Build a deterministic EEGNet module. Input x: (B, 1, n_chans, n_times). Lazy torch."""
    import torch
    import torch.nn as nn

    F2 = F1 * D

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
                nn.BatchNorm2d(F1))
            self.depthwise = nn.Sequential(
                nn.Conv2d(F1, F2, (n_chans, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.separable = nn.Sequential(
                nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
                nn.Conv2d(F2, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
            with torch.no_grad():
                feat = self._features(torch.zeros(1, 1, n_chans, n_times))
            self.embedding_dim = int(feat.shape[1])
            self.classifier = nn.Linear(self.embedding_dim, n_classes)

        def _features(self, x):
            x = self.block1(x)
            x = self.depthwise(x)
            x = self.separable(x)
            return torch.flatten(x, 1)

        def encode(self, x):
            return self._features(x)

        def forward(self, x):
            return self.classifier(self._features(x))

    return EEGNet()


# ---- canonical, pickle-free state serialization (deterministic bytes) --------------------------------------------------------
_MAGIC = b"ACARV5SD"


def canonical_state_bytes(state_dict):
    """Serialize a torch state_dict to DETERMINISTIC, pickle-free bytes: magic + json header (sorted names, dtypes, shapes) +
    concatenated little-endian float32 tensor bytes. Reproducible across runs (no zip timestamps, no pickle)."""
    import numpy as np
    names = sorted(state_dict.keys())
    header, blobs = [], []
    for name in names:
        arr = state_dict[name].detach().cpu().numpy().astype("<f4", copy=False)
        header.append({"name": name, "shape": list(arr.shape)})
        blobs.append(np.ascontiguousarray(arr).tobytes())
    hjson = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out = bytearray(_MAGIC)
    out += struct.pack("<I", len(hjson))
    out += hjson
    for b in blobs:
        out += b
    return bytes(out)


def load_state_arrays(blob):
    """Inverse of canonical_state_bytes → {name: numpy float32 array}. numpy lazy."""
    import numpy as np
    if bytes(blob[:len(_MAGIC)]) != _MAGIC:
        raise ValueError("bad canonical state blob (magic mismatch)")
    off = len(_MAGIC)
    (hlen,) = struct.unpack("<I", blob[off:off + 4])
    off += 4
    header = json.loads(bytes(blob[off:off + hlen]).decode("utf-8"))
    off += hlen
    out = {}
    for entry in header:
        shape = tuple(entry["shape"])
        n = 1
        for s in shape:
            n *= s
        arr = np.frombuffer(blob[off:off + 4 * n], dtype="<f4").reshape(shape).copy()
        off += 4 * n
        out[entry["name"]] = arr
    return out


def load_state_into(module, blob):
    """Load canonical bytes into a torch module's state_dict (float32), in-place."""
    import torch
    arrays = load_state_arrays(blob)
    sd = module.state_dict()
    if set(arrays) != set(sd):
        raise ValueError("canonical state blob keys do not match the module state_dict")
    module.load_state_dict({k: torch.from_numpy(arrays[k]).to(sd[k].dtype).reshape(sd[k].shape) for k in arrays})
    return module
