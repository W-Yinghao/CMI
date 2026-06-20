"""Self-supervised / reconstruction auxiliaries to prevent class-latent collapse.

This module addresses the review concern (5.2 / 5.6) that the class-latent ``z_c``
can collapse to a trivial representation when the only training signal is a
class-conditional density / CMI objective.  We add two complementary, *decoupled*
auxiliaries that operate **only** on the produced latent ``z_c`` and the raw input
``x`` -- they never touch encoder internals, so they can be bolted onto any encoder:

1. :func:`vicreg_penalty` -- a variance + covariance regulariser (VICReg, Bardes
   et al. 2022) that explicitly pushes each latent dimension to retain spread and
   to be decorrelated, directly fighting dimensional collapse.

2. :class:`SSLAux` -- a small linear-ish decoder that performs *masked
   reconstruction* of a temporally-pooled view of the input from ``z_c``.  Forcing
   ``z_c`` to in-paint masked-out channels guarantees it must encode
   non-degenerate information about the signal.

Only ``torch`` / ``torch.nn`` / ``torch.nn.functional`` / ``numpy`` are used.
"""
from __future__ import annotations

import math

import numpy as np  # noqa: F401  (allowed dependency; kept for parity / downstream use)
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["vicreg_penalty", "SSLAux"]


def vicreg_penalty(
    z: torch.Tensor,
    std_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """VICReg variance + covariance regularisation on a latent batch.

    Parameters
    ----------
    z : torch.Tensor
        Latent batch of shape ``[N, d]``.
    std_target : float
        Target per-dimension standard deviation (hinge target).
    eps : float
        Numerical stabiliser added under the square-root.

    Returns
    -------
    (total, info)
        ``total`` is the differentiable scalar ``var_term + cov_term``; ``info``
        carries the two component values as python floats.

    Notes
    -----
    * variance term  = mean over dims of ``relu(std_target - sqrt(Var_j + eps))``
      (Var_j is the unbiased per-dimension variance).
    * covariance term = sum of squared off-diagonal entries of the covariance of
      the centred latents, divided by ``d``.

    The penalty is guarded for ``N < 2`` (variance is undefined): it returns a
    zero tensor still attached to ``z`` so gradients/`.backward()` stay valid.
    """
    if z.dim() != 2:
        raise ValueError(f"vicreg_penalty expects a 2-D [N, d] tensor, got shape {tuple(z.shape)}")

    n, d = z.shape

    if n < 2:
        # Variance/covariance undefined for a single sample: emit a structural
        # zero that is still differentiable w.r.t. z (keeps the graph intact).
        zero = (z.sum() * 0.0)
        return zero, {"vicreg_var": 0.0, "vicreg_cov": 0.0}

    # ---- variance term -------------------------------------------------------
    # Unbiased variance (matches the VICReg paper which uses ddof=1 / N-1).
    var = z.var(dim=0, unbiased=True)  # [d]
    std = torch.sqrt(var + eps)
    var_term = torch.mean(F.relu(std_target - std))

    # ---- covariance term -----------------------------------------------------
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.t() @ z_centered) / (n - 1)  # [d, d]
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_term = off_diag.pow(2).sum() / d

    total = var_term + cov_term
    info = {
        "vicreg_var": float(var_term.detach().cpu().item()),
        "vicreg_cov": float(cov_term.detach().cpu().item()),
    }
    return total, info


class SSLAux(nn.Module):
    """Decoupled self-supervised auxiliary head (masked recon + VICReg).

    Operates only on ``(z_c, x)``: it reconstructs a temporally average-pooled
    view of the raw input ``x`` from the class latent ``z_c``, optionally with a
    Bernoulli channel-mask so that ``z_c`` must *in-paint* the missing entries.
    A VICReg penalty on ``z_c`` can be added to fight dimensional collapse.

    Parameters
    ----------
    z_c_dim : int
        Dimensionality of the class latent ``z_c``.
    n_chans : int
        Number of EEG channels of the raw input ``x``.
    n_times : int
        Number of time samples of the raw input ``x``.
    recon_pool : int
        Temporal pooling factor; the reconstruction target has
        ``T_pool = ceil(n_times / recon_pool)`` time bins.
    mask_ratio : float
        Fraction of pooled target entries to mask out (only used when
        ``masked_recon`` is True).
    vicreg : bool
        Whether to add the VICReg penalty on ``z_c``.
    masked_recon : bool
        If True, the MSE is computed only on the masked-out entries; if False,
        on all entries.
    """

    def __init__(
        self,
        z_c_dim: int,
        n_chans: int,
        n_times: int,
        recon_pool: int = 8,
        mask_ratio: float = 0.3,
        vicreg: bool = True,
        masked_recon: bool = True,
    ) -> None:
        super().__init__()
        self.z_c_dim = int(z_c_dim)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.recon_pool = int(recon_pool)
        self.mask_ratio = float(mask_ratio)
        self.use_vicreg = bool(vicreg)
        self.masked_recon = bool(masked_recon)

        # Pooled temporal resolution; adaptive_avg_pool1d handles any n_times.
        self.t_pool = int(math.ceil(self.n_times / max(1, self.recon_pool)))
        self.out_dim = self.n_chans * self.t_pool

        self.decoder = nn.Sequential(
            nn.Linear(self.z_c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim),
        )

    def _pooled_target(self, x: torch.Tensor) -> torch.Tensor:
        """Average-pool ``x`` [N, n_chans, n_times] -> flattened [N, n_chans*T_pool]."""
        # adaptive_avg_pool1d pools the last (time) dim to exactly t_pool bins.
        pooled = F.adaptive_avg_pool1d(x, self.t_pool)  # [N, n_chans, t_pool]
        return pooled.reshape(pooled.shape[0], -1)  # [N, n_chans*t_pool]

    def forward(self, z_c: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the auxiliary loss.

        Returns ``(total, info)`` where ``total`` is the differentiable scalar
        ``recon_mse + vicreg_total`` and ``info`` reports each component as a
        python float.
        """
        device = z_c.device
        target = self._pooled_target(x).to(device)  # [N, out_dim]
        recon = self.decoder(z_c)  # [N, out_dim]

        if self.masked_recon:
            # Bernoulli mask: 1 == masked-out entry whose value must be inferred.
            mask = (torch.rand(target.shape, device=device) < self.mask_ratio)
            n_masked = mask.sum()
            if n_masked > 0:
                diff = (recon - target) ** 2
                recon_mse = diff[mask].mean()
            else:
                # No entry was masked (e.g. mask_ratio==0 or unlucky tiny batch):
                # fall back to full MSE so the loss stays well-defined & finite.
                recon_mse = F.mse_loss(recon, target)
        else:
            recon_mse = F.mse_loss(recon, target)

        info: dict = {"recon_mse": float(recon_mse.detach().cpu().item())}

        total = recon_mse
        if self.use_vicreg:
            vic_total, vic_info = vicreg_penalty(z_c)
            total = total + vic_total
            info["vicreg_var"] = vic_info["vicreg_var"]
            info["vicreg_cov"] = vic_info["vicreg_cov"]
            info["vicreg_total"] = vic_info["vicreg_var"] + vic_info["vicreg_cov"]

        info["total"] = float(total.detach().cpu().item())
        return total, info


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- SSLAux end-to-end self-test ----------------------------------------
    aux = SSLAux(32, 16, 256)
    z_c = torch.randn(8, 32, requires_grad=True)
    x = torch.randn(8, 16, 256)

    loss, info = aux.forward(z_c, x)
    assert loss.dim() == 0, f"loss must be scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    print("SSLAux forward OK -> total =", float(loss.item()), "components:", info)

    loss.backward()
    assert z_c.grad is not None, "z_c.grad not populated by backward()"
    assert torch.isfinite(z_c.grad).all(), "z_c.grad contains non-finite values"
    print("SSLAux backward OK -> z_c.grad norm =", float(z_c.grad.norm().item()))

    # ---- vicreg_penalty standalone self-test --------------------------------
    z = torch.randn(8, 32, requires_grad=True)
    v_total, v_info = vicreg_penalty(z)
    assert v_total.dim() == 0 and torch.isfinite(v_total), "vicreg total invalid"
    v_total.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all(), "vicreg grad invalid"
    print("vicreg_penalty OK -> total =", float(v_total.item()), "info:", v_info)

    # ---- tiny-batch (N<2) robustness ----------------------------------------
    z1 = torch.randn(1, 32, requires_grad=True)
    g_total, g_info = vicreg_penalty(z1)
    assert torch.isfinite(g_total) and float(g_total.item()) == 0.0, "N<2 guard failed"
    g_total.backward()  # must not raise
    print("vicreg_penalty N<2 guard OK -> total =", float(g_total.item()), "info:", g_info)

    # masked_recon=False variant
    aux_full = SSLAux(32, 16, 250, masked_recon=False)  # non-multiple n_times
    z_c2 = torch.randn(4, 32, requires_grad=True)
    x2 = torch.randn(4, 16, 250)
    loss2, info2 = aux_full.forward(z_c2, x2)
    assert torch.isfinite(loss2)
    loss2.backward()
    assert z_c2.grad is not None
    print("SSLAux(masked_recon=False, n_times=250) OK -> total =", float(loss2.item()))

    print("ALL SELF-TESTS PASSED")
