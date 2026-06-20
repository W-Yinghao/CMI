"""Task / nuisance latent disentanglement penalties (review section 5.6).

This module implements cheap, fully-differentiable surrogates for the composite
disentanglement objective

    L_disentangle = I(Z_c ; Z_n | Y)  +  eta * I(Z_n ; Y | D)  -  kappa * I(Z_n ; D)

where ``Z_c`` is the *task* latent, ``Z_n`` the *nuisance* latent, ``Y`` the label
and ``D`` the domain.  We deliberately AVOID neural mutual-information estimators
(MINE/CLUB style) which are noisy and expensive.  Instead each information term is
replaced by a lightweight statistical surrogate:

  * ``I(Z_c ; Z_n | Y)``  -> class-conditional HSIC (or cross-covariance) between the
    two latents, computed within each class and averaged.  Zero HSIC <=> the kernel
    embeddings are uncorrelated, a necessary condition for independence.
  * ``I(Z_n ; Y | D)``    -> reduction in label cross-entropy achieved by a small
    probe reading ``z_n``; we penalise any label information leaking into the
    nuisance branch.
  * ``I(Z_n ; D)``        -> negative cross-entropy of a small domain probe reading
    ``z_n``; we REWARD the nuisance branch carrying domain information.

Only ``torch``, ``torch.nn``, ``torch.nn.functional`` and ``numpy`` are used.  The
module is self-contained and does not import (or mutate) the sibling ``cmi`` package.

Run ``python -m h2cmi.disentangle.penalties`` to execute the self-test.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept available per package conventions / future use)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------
def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    """Return the [N, N] matrix of squared Euclidean distances between rows of x."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b ; clamp to kill tiny negatives.
    sq_norms = (x * x).sum(dim=1, keepdim=True)  # [N, 1]
    d2 = sq_norms + sq_norms.t() - 2.0 * (x @ x.t())
    return d2.clamp_min(0.0)


def _median_sigma(d2: torch.Tensor) -> float:
    """Median pairwise *squared* distance heuristic, guarded against 0."""
    # Use the full matrix (incl. zero diagonal) median, matching the spec wording
    # "median of pairwise squared distances".  Guard against a degenerate 0 median.
    med = torch.median(d2.detach())
    sigma = float(med.item())
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    return sigma


def _rbf_kernel(x: torch.Tensor, sigma: float | None) -> torch.Tensor:
    """RBF (Gaussian) kernel matrix K_ij = exp(-||x_i - x_j||^2 / (2 sigma))."""
    d2 = _pairwise_sq_dists(x)
    if sigma is None:
        sigma = _median_sigma(d2)
    sigma = max(float(sigma), 1e-12)
    return torch.exp(-d2 / (2.0 * sigma))


def hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> torch.Tensor:
    """Biased empirical HSIC with RBF kernels.

    Parameters
    ----------
    x : torch.Tensor, shape [N, dx]
    y : torch.Tensor, shape [N, dy]
    sigma_x, sigma_y : optional float kernel bandwidths.  ``None`` -> median
        pairwise-squared-distance heuristic.

    Returns
    -------
    torch.Tensor
        Scalar, differentiable: ``trace(Kx_c @ Ky_c) / (N - 1)^2`` where ``Kx_c``,
        ``Ky_c`` are the centred kernel matrices (H = I - 1/N).
    """
    if x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() != 2:
        y = y.reshape(y.shape[0], -1)
    n = x.shape[0]
    if n < 2:
        # (N - 1)^2 would be 0; HSIC is undefined for <2 samples.
        return x.sum() * 0.0

    kx = _rbf_kernel(x, sigma_x)
    ky = _rbf_kernel(y, sigma_y)

    h = torch.eye(n, dtype=kx.dtype, device=kx.device) - (1.0 / n)
    kx_c = h @ kx @ h
    ky_c = h @ ky @ h

    # trace(A @ B) == sum(A * B^T); both are symmetric here.
    hsic_val = (kx_c * ky_c).sum() / float((n - 1) ** 2)
    return hsic_val


def cross_covariance_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared cross-covariance between centred ``x`` and ``y``.

    Returns ``||Cov(x, y)||_F^2 / (dx * dy)``, differentiable.
    """
    if x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() != 2:
        y = y.reshape(y.shape[0], -1)
    n = x.shape[0]
    dx = x.shape[1]
    dy = y.shape[1]
    if n < 2 or dx == 0 or dy == 0:
        return x.sum() * 0.0

    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    cov = (xc.t() @ yc) / float(n - 1)  # [dx, dy]
    return (cov * cov).sum() / float(dx * dy)


def orthogonality_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared cosine-like alignment between row-normalised ``x`` and ``y``.

    Returns ``(1/N) sum_i <x_i_hat, y_i_hat>^2`` with unit-normalised rows,
    differentiable.  Requires ``x`` and ``y`` to share the same feature dimension.
    """
    if x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() != 2:
        y = y.reshape(y.shape[0], -1)
    n = x.shape[0]
    if n == 0:
        return x.sum() * 0.0
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "orthogonality_penalty requires matching feature dims, "
            f"got x[{x.shape[1]}] vs y[{y.shape[1]}]"
        )

    xh = F.normalize(x, p=2.0, dim=1, eps=1e-8)
    yh = F.normalize(y, p=2.0, dim=1, eps=1e-8)
    align = (xh * yh).sum(dim=1)  # [N] per-row dot product of unit vectors
    return (align * align).mean()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
def _mlp_probe(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    """A small 2-layer MLP probe."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class DisentangleLoss(nn.Module):
    """Composite task/nuisance disentanglement loss (review section 5.6).

    Minimising the returned scalar:

      * drives the class-conditional dependence between ``z_c`` and ``z_n`` to 0
        (surrogate for ``I(Z_c ; Z_n | Y) -> 0``),
      * removes any label information from the nuisance latent
        (surrogate for ``I(Z_n ; Y | D) -> 0``, weighted by ``eta``),
      * encourages the nuisance latent to carry domain information
        (surrogate for ``I(Z_n ; D)`` *up*, weighted by ``kappa``).

    The two probes are trained by the SAME cross-entropies that appear in the loss
    (a deliberate min-min surrogate): there is no adversarial inner loop, so the
    objective stays cheap and stable.
    """

    def __init__(
        self,
        z_c_dim: int,
        z_n_dim: int,
        n_classes: int,
        n_domains: int,
        eta: float = 0.1,
        kappa: float = 0.1,
        method: str = "hsic",
    ) -> None:
        super().__init__()
        if method not in ("hsic", "cross_cov"):
            raise ValueError(
                f"method must be 'hsic' or 'cross_cov', got {method!r}"
            )
        self.z_c_dim = int(z_c_dim)
        self.z_n_dim = int(z_n_dim)
        self.n_classes = int(n_classes)
        self.n_domains = int(n_domains)
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.method = method

        # Probe estimating how well z_n predicts the DOMAIN (I(Z_n;D)).
        self.dom_head = _mlp_probe(self.z_n_dim, self.n_domains)
        # Probe estimating label leakage into z_n (I(Z_n;Y)).
        self.task_head_zn = _mlp_probe(self.z_n_dim, self.n_classes)

        # Constant reference: entropy of a uniform class prior.
        self.register_buffer(
            "_h_class_prior",
            torch.tensor(float(np.log(max(self.n_classes, 1)))),
        )

    # ----- helpers ---------------------------------------------------------
    def _dependence(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.method == "cross_cov":
            return cross_covariance_penalty(a, b)
        return hsic(a, b)

    def _term_zc_zn(self, z_c: torch.Tensor, z_n: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Class-averaged within-class dependence of z_c and z_n."""
        zero = z_c.sum() * 0.0
        if y.numel() == 0:
            return zero
        terms = []
        for c in torch.unique(y):
            mask = y == c
            if int(mask.sum().item()) < 4:  # skip tiny class groups
                continue
            terms.append(self._dependence(z_c[mask], z_n[mask]))
        if not terms:
            return zero
        return torch.stack(terms).mean()

    # ----- forward ---------------------------------------------------------
    def forward(
        self,
        z_c: torch.Tensor,
        z_n: torch.Tensor,
        y: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        y = y.long()
        d = d.long()

        # term 1: I(Z_c ; Z_n | Y)  -- MINIMISED (positive sign).
        term_zc_zn = self._term_zc_zn(z_c, z_n, y)

        # term 2: label info leaking into z_n  -- penalise (sign +eta).
        # reduction in label uncertainty = H_prior - CE(probe(z_n), y), relu'd.
        if y.numel() > 0:
            logits_y = self.task_head_zn(z_n)
            ce_y = F.cross_entropy(logits_y, y)
        else:
            ce_y = z_n.sum() * 0.0
        term_zn_y = F.relu(self._h_class_prior - ce_y)

        # term 3: domain info in z_n  -- REWARD (minimising +kappa*CE pushes CE
        # down, which raises I(Z_n;D); the dom_head is trained by the same CE).
        if d.numel() > 0:
            logits_d = self.dom_head(z_n)
            ce_dom = F.cross_entropy(logits_d, d)
        else:
            ce_dom = z_n.sum() * 0.0

        total = term_zc_zn + self.eta * term_zn_y + self.kappa * ce_dom

        info = {
            "term_zc_zn": float(term_zc_zn.detach().item()),
            "term_zn_y": float(term_zn_y.detach().item()),
            "ce_dom": float(ce_dom.detach().item()),
            "ce_y": float(ce_y.detach().item()) if torch.is_tensor(ce_y) else float(ce_y),
            "total": float(total.detach().item()),
        }
        return total, info


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)

    # --- free functions ---------------------------------------------------
    x = torch.randn(50, 8, requires_grad=True)
    y = torch.randn(50, 5, requires_grad=True)
    z = torch.randn(50, 8, requires_grad=True)

    h_default = hsic(x, y)
    h_fixed = hsic(x, y, sigma_x=2.0, sigma_y=3.0)
    cc = cross_covariance_penalty(x, y)
    orth = orthogonality_penalty(x, z)  # same feature dim
    for name, val in [("hsic", h_default), ("hsic_fixed", h_fixed),
                      ("cross_cov", cc), ("orthogonality", orth)]:
        assert val.dim() == 0, f"{name} not scalar"
        assert torch.isfinite(val), f"{name} not finite: {val}"
    assert h_default.item() >= -1e-6, "biased HSIC should be ~non-negative"
    assert cc.item() >= 0.0 and orth.item() >= 0.0
    # gradients flow
    (h_default + cc + orth + h_fixed).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[ok] free functions: "
          f"hsic={h_default.item():.4e} hsic_fixed={h_fixed.item():.4e} "
          f"cross_cov={cc.item():.4e} orth={orth.item():.4f}")

    # --- DisentangleLoss (hsic) ------------------------------------------
    loss_mod = DisentangleLoss(32, 16, 3, 5, eta=0.1, kappa=0.1, method="hsic")
    z_c = torch.randn(64, 32, requires_grad=True)
    z_n = torch.randn(64, 16, requires_grad=True)
    yb = torch.randint(0, 3, (64,))
    db = torch.randint(0, 5, (64,))

    total, info = loss_mod(z_c, z_n, yb, db)
    assert total.dim() == 0 and torch.isfinite(total), f"bad total: {total}"
    assert set(info) >= {"term_zc_zn", "term_zn_y", "ce_dom", "total"}
    for k, v in info.items():
        assert np.isfinite(v), f"info[{k}] not finite: {v}"
    total.backward()
    assert z_c.grad is not None and torch.isfinite(z_c.grad).all()
    assert z_n.grad is not None and torch.isfinite(z_n.grad).all()
    # probe parameters also receive gradient (min-min surrogate)
    probe_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in loss_mod.parameters()
    )
    assert probe_grad, "probe parameters got no gradient"
    print("[ok] DisentangleLoss(hsic): total={total:.4e} info={info}".format(
        total=total.item(), info=info))

    # --- DisentangleLoss (cross_cov) + robustness to a tiny batch --------
    loss_cc = DisentangleLoss(32, 16, 3, 5, method="cross_cov")
    tot2, info2 = loss_cc(torch.randn(64, 32), torch.randn(64, 16), yb, db)
    assert torch.isfinite(tot2)
    tot2.backward()  # no requires_grad inputs, but probes should still get grad
    print(f"[ok] DisentangleLoss(cross_cov): total={tot2.item():.4e}")

    # tiny batch: all classes have <4 samples -> term_zc_zn must be 0, finite total
    z_c_t = torch.randn(3, 32, requires_grad=True)
    z_n_t = torch.randn(3, 16, requires_grad=True)
    y_t = torch.tensor([0, 1, 2])
    d_t = torch.tensor([0, 1, 2])
    tot3, info3 = loss_mod(z_c_t, z_n_t, y_t, d_t)
    assert torch.isfinite(tot3), f"tiny-batch total not finite: {tot3}"
    assert info3["term_zc_zn"] == 0.0, "tiny-batch zc_zn should be 0"
    tot3.backward()
    print(f"[ok] tiny-batch robust: total={tot3.item():.4e} term_zc_zn={info3['term_zc_zn']}")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
