# Integrating TOS-CMI with the real EEG stack

The package is self-contained on the synthetic. To run on EEG it needs (a) a
representation `Z` and (b) class/domain labels `(y, d)` per trial. Everything below
points at code that already exists in `cmi/` and `repos/TSMNet/`; nothing here mutates
those packages.

The trainer discipline is two-step, matching `cmi/train/trainer.py`:

```
sp = SelectivePenalty(z_dim, n_cls, n_dom, priors=(y_all, d_all),
                      pcfg=PenaltyConfig(lam=λ, refresh_every=K, prior_mode="subject"))

# every K epochs, on a representative batch (no grad):
sp.refresh(Z_epoch.detach(), y_epoch, d_epoch)

# Step A (detached Z): fit the conditional-domain critic on P_N Z
loss_A = sp.posterior_loss(Z.detach(), y, d)

# Step B (grad Z): task CE + selective penalty (critic frozen)
loss_B = F.cross_entropy(logits, y) + sp.penalty(Z, y, d)
```

`sp.penalty` is exactly 0 whenever the selector returns identity, so a fold where no safe
nuisance subspace exists trains as plain ERM — by design.

## 1. TSMNet / SPD (the primary counterexample)

`cmi/models/backbones.py: TSMNetBackbone` wraps `repos/TSMNet/spdnets/models/tsmnet.py`
(`BiMap → ReEig → SPD-BN → LogEig → classifier`). Global LPC is applied to the LogEig
tangent vector `Z` and collapses the SPD geometry. Two attach points:

* **tangent `Z` (default).** `forward(x, return_latent=True)` returns the LogEig vector.
  Run the selector on it; `P_N` lives in tangent space. Lowest-friction starting point.
* **pre/post-BN SPD matrices.** `forward(..., return_prebn=True, return_postbn=True)`
  returns the `[B,1,s,s]` SPD matrices. Flatten the upper triangle (or use the tangent
  map) to get a vector for the Fishers; the selected `P_N` then identifies *which SPD
  eigen-directions* are domain-rich/label-light. This is the version that can plausibly
  survive where global LPC collapses, because it never touches the task-entangled
  directions.

**Pass/fail:** TOS-CMI must beat global `lpc_prior` on TSMNet LOSO (2a) at matched
source-only-selected λ, *and* the selected subspace must be stable across seeds
(`eval.stability`). If it can't save the clearest collapse case, stop (THEORY §6).

## 2. λ on 2a (the over-erasure counterexample)

`cmi/data/moabb_data.py: load("BNCI2014_001")` + `domain_labels(meta, "subject")` +
`loso_splits(meta)`. The claim to test: where global LPC needs a small, fragile λ to
avoid eating labels, the selective penalty is **λ-robust** because it only acts on
label-light directions. Sweep λ for both; TOS-CMI's accuracy should be flat where
global LPC's drops. Select λ source-only with `cmi/run_lambda_select.py`'s protocol (no
target labels).

## 3. GraphCMI (the uneven-leakage counterexample)

`cmi/models/gnn.py: GraphCMINet.forward_graph(x)` exposes `node_Z [B,C,hidden]` and
`edge_logits [B,C,C]`. Leakage is concentrated in some channels/nodes. Run the same
machinery **per node** (one `SubspaceSelector` over the node-feature dim, or stack nodes
and let the projector pick node-blocks) so the CMI budget is spent where the leakage
actually is. `cmi/methods/graph_regularizers.py: NodePosterior/EdgePosterior` already
give the per-node/edge `I(Z_v;D|Y)` term to apply on the selected node directions.

## 4. Reusing the existing critic instead of the local one

`SelectivePenalty` ships its own `ConditionalDomainCritic` so the package runs without the
MOABB/braindecode import chain. Inside `cmi/train/trainer.py` you may instead reuse
`cmi.methods.regularizers.DomainPosteriors` directly on the **projected** features:

```
post = DomainPosteriors(z_dim, n_dom, n_cls, empirical_priors(y, d, n_dom, n_cls))
Zn = selector.project(z)                 # P_N z
loss_A += post.posterior_loss(Zn.detach(), yb, db)
loss_B += λ * post.reg("lpc_prior", Zn, yb)     # I(P_N Z; D|Y)
```

This makes TOS-CMI a one-line wrapper (`z → P_N z`) around the validated AAAI penalty,
which is the cleanest ablation: identical estimator, the only change is *where* it acts.

## 5. Register as a trainer method

Add `"tos_cmi"` to `ALL_METHODS` in `cmi/train/trainer.py`, build a `SubspaceSelector`
after the backbone, refresh it every `refresh_every` epochs on a held batch, and in the
loss-assembly block (lines ~335-339) add `loss += lam_t * selector_penalty`. Diagnostics
to log per fold: `selector.summary()` (k, selected ratios, label shares, null floor), the
stability overlap across seeds, and the leakage probes from `cmi/eval/metrics.py`
(`leakage_probe`, `decoder_leakage_probe`) on `P_N Z` vs `(I−P_N) Z`.

## Confirmatory protocol (real-EEG, honest)

Unified preprocessing (memory: raw-signal for leakage, MOABB 250→128 Hz), LOSO × 5–10
seeds, subject-clustered inference, source-only λ selection, and report **both** the
accuracy panel and the selection-stability gate. A positive result is: matched-λ
accuracy ≥ global LPC on TSMNet **and** stable selection **and** λ-robustness on 2a.
Anything less is "a more complicated regularizer" — the stated stop condition.
