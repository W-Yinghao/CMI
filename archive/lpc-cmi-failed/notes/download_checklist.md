# Tri-CMI — papers & repos to obtain
_Auto-generated from the carrier-exploration workflow (w7a3odm0r)._

## Papers (arXiv id / URL)
- **CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information** — `2006.12013`
  - Direct conceptual parent of LPC-CMI; CLUBForCategorical is the reusable q_psi(D|Z) building block (continuous Z, discrete D). Upper-bound direction is what justifies minimization. Must-cite estimator basis.
- **The IM Algorithm: A Variational Approach to Information Maximization (Barber & Agakov, NeurIPS 2003)** — `https://proceedings.neurips.cc/paper/2003/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html`
  - Mathematical foundation: I(Z;D|Y)=H(D|Y)-H(D|Z,Y) for discrete D shows our posterior-KL IS a principled MI bound and explains Y-protection via conditioning.
- **Formal Limitations on the Measurement of Mutual Information (McAllester & Stratos, AISTATS 2020)** — `1811.04251`
  - Must-cite justification for choosing an UPPER-bound posterior-KL over lower-bound estimators (MINE/InfoNCE/DV are statistically capped at O(ln N) and high-variance) — core to the methodological argument.
- **FedSR: A Simple and Effective Domain Generalization Method for Federated Learning** — `https://proceedings.neurips.cc/paper_files/paper/2022/hash/fd946a6c99541fddc3d64a3ea39a1bc2-Abstract-Conference.html`
  - Closest published precedent to 'posterior-KL to a label-conditional reference'; engineering template (~40-line closed-form per-class KL). Position as analogous-but-distinct (it bounds I(Z;X|Y), not I(Z;D|Y)).
- **Invariant Representations without Adversarial Training (Moyer et al., NeurIPS 2018)** — `1805.09458`
  - Cleanest theoretical precedent that conditional invariance via a single variational/posterior-KL (no adversary) protects Y — the design stance of Tri-CMI.
- **Efficient Conditionally Invariant Representation Learning (CIRCE, ICLR 2023 Oral)** — `2212.08645`
  - Best kernel-side competitor: a plug-in conditional-independence regularizer Z _||_ D | Y that protects Y; official MIT PyTorch. Principal kernel ablation/alternative carrier.
- **Conditional Adversarial Domain Adaptation (CDAN, NeurIPS 2018)** — `1705.10667`
  - Closest adversarial prior-art for 'do not erase Y' (f-tensor-g multilinear conditioning). Baseline to position against.
- **Deep Domain Generalization via Conditional Invariant Adversarial Networks (CDANN, ECCV 2018)** — `https://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf`
  - THE single closest comparison: conditions discriminator on Y and pushes toward UNIFORM domain posterior; isolates 'uniform target vs pi_y(D) target' and 'adversarial vs adversary-free KL'.
- **Invariant Information Bottleneck for Domain Generalization (IIB, AAAI 2022)** — `2106.06333`
  - Theoretical foil: minimizes I(Y;D|Z) (reversed variable ordering) + I(X;Z) compression. Headline distinction in related-work + an ablation (swap our term for theirs).
- **EEG-DG: A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification** — `2311.05415`
  - Most-related application baseline: same MI-EEG calibration-free DG task, same datasets (2a/2b/OpenBMI), reusable Modified-EEGNet carrier; head-to-head accuracy/kappa to beat.
- **SPD domain-specific batch normalization (TSMNet/SPDDSMBN)** — `2206.01323`
  - Strongest geometric DG baseline on 6 MOABB datasets in our LOSO/cross-session regime; removes D marginally/implicitly — motivating contrast for label-conditional removal.
- **EEGNet: a compact convolutional neural network for EEG-based BCIs** — `1611.08024`
  - Primary carrier reference (braindecode EEGNetv4).
- **Deep learning with convolutional neural networks for EEG decoding and visualization (Schirrmeister et al.)** — `1703.05051`
  - ShallowFBCSPNet + Deep4Net carrier reference; canonical MI baselines.
- **EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization (IEEE TNSRE 2023)** — `https://ieeexplore.ieee.org/document/9991178`
  - Transformer carrier for architecture-agnosticism (braindecode EEGConformer, return_features verified).
- **Conditional Mutual Information Estimation for Mixed, Discrete and Continuous Data (knncmi)** — `1912.03387`
  - Trusted offline mixed-data CMI ground-truth (Z continuous, D,Y discrete) for Protocol-D validation of the LPC-CMI proxy.
- **In Search of Lost Domain Generalization (DomainBed, ICLR 2021)** — `2007.01434`
  - Source of the canonical DANN/CDANN/CORAL/MMD/IRM/GroupDRO baseline implementations and the model-selection/sweep harness template.

## Repos
- **braindecode** (OFFICIAL) — https://github.com/braindecode/braindecode
  - PRIMARY CARRIER LIBRARY (already installed v0.8 in env 'icml'). Provides EEGNetv4, ShallowFBCSPNet, Deep4Net, EEGConformer (return_features=True verified), ATCNet — all import+forward-pass confirmed. BSD-3. NOTE: MSVTNet and Labram are NOT in 0.8 (need >=1.4); do not rely on them under the pinned stack.
- **moabb** (OFFICIAL) — https://github.com/NeuroTechX/moabb
  - Dataset access + LOSO/cross-session/cross-dataset evaluation splits (installed v1.2). BSD-3. Datasets already cached offline under /projects/EEG-foundation-model/datalake/raw.
- **Linear95/CLUB** (OFFICIAL) — https://github.com/Linear95/CLUB
  - Reference for CLUBForCategorical (our q_psi backbone) + MINE/NWJ/InfoNCE/CLUBSample probes, plus MI_DA/MI_IB usage examples. NO LICENSE -> READ ONLY; reimplement the ~25-line class in our own MIT code, do not vendor.
- **DomainBed** (OFFICIAL) — https://github.com/facebookresearch/DomainBed
  - MIT. Verbatim-usable DANN, CDANN (closest comparison), CORAL, MMD, CausIRL, IRM, GroupDRO, VREx penalty classes + model-selection harness. Lift penalty fns onto braindecode backbones; replace image data loaders.
- **namratadeka/circe** (OFFICIAL) — https://github.com/namratadeka/circe
  - MIT. Official PyTorch CIRCE conditional-independence regularizer (Z _||_ D | Y) — principal kernel-side competitor and alternative carrier; HSCIC/GCM baselines included. Adapt Y->D conditional-mean regression for one-hot/discrete Y.
- **XC-ZhongHIT/EEG-DG** (OFFICIAL) — https://github.com/XC-ZhongHIT/EEG-DG
  - Head-to-head MI-EEG calibration-free DG baseline on BCI-IV-2a/2b + OpenBMI; reusable Modified-EEGNet backbone (mmd.py, Dist_Loss.py). NO LICENSE -> contact authors before vendoring; reproduce numbers, reimplement losses.
- **hongyizhi/SCLDGN** (OFFICIAL) — https://github.com/hongyizhi/SCLDGN
  - MI-specific calibration-free LOSO DG; ships clean droppable loss modules (lossFunction/scl.py SupConLoss w/ domain arg, coral.py, mmd.py, irm.py, mcc.py). NO LICENSE -> reimplement standard SupCon ourselves; use as label-protecting contrastive auxiliary/baseline.
- **Luodian/IIB** (OFFICIAL) — https://github.com/Luodian/IIB
  - MIT. Official IIB (minimizes I(Y;D|Z)+I(X;Z)) on DomainBed — theoretical foil with reversed variable ordering; swap-in ablation and per-environment variational-bound design reference.
- **rkobler/TSMNet** (OFFICIAL) — https://github.com/rkobler/TSMNet
  - BSD-3. Strongest geometric DG baseline (SPDDSMBN per-domain SPD whitening) validated on 6 MOABB datasets; built on geoopt 0.5.x + skorch + moabb (stack-compatible). Headline geometric comparator.
- **pyRiemann** (OFFICIAL) — https://github.com/pyRiemann/pyRiemann
  - BSD-3 (installed v0.7). Covariances + TangentSpace + per-domain recentering for the optional deterministic geometric carrier Z and post-hoc leakage probe substrate.
- **omesner/knncmi** (OFFICIAL) — https://github.com/omesner/knncmi
  - GPL-3.0. Mesner-Shalizi mixed kNN CMI — trusted OFFLINE ground-truth checker for I(Z;D|Y) on Protocol-D synthetics. Keep as an isolated eval-only dependency (GPL), never vendored into the release.
- **yi-ding-cs/EEG-Deformer** (OFFICIAL) — https://github.com/yi-ding-cs/EEG-Deformer
  - OPTIONAL extra architecture-diversity carrier (standalone PyTorch nn.Module). CBCR NON-COMMERCIAL license -> do NOT redistribute; reproduce-it-yourself probe only, lowest priority (not MI-validated, not in braindecode).

## Recommended carrier stack
- PRIMARY ENCODER: braindecode 0.8 EEGNetv4 (verified import+forward in env 'icml': EEGNetv4(n_chans=22,n_outputs=4,n_times=1000)->[B,4]). Low-dim F2 bottleneck (default 16; bump to F2=32) is the clean continuous Z for q_psi; tiny capacity limits subject memorization. Tap Z via forward hook on the layer before final conv/flatten.
- PRIMARY ENCODER #2 (canonical MI baseline): braindecode ShallowFBCSPNet (~46k params, official trialwise recipe AdamW lr=0.0625*0.01 bs=64). Flattened pre-classifier map needs a small global-pool/projection head to make a low-dim Z for q_psi.
- ARCHITECTURE-AGNOSTICISM PROOF: braindecode EEGConformer with return_features=True (VERIFIED: returns (logits[B,4], features[B,32]) directly — the 32-dim token IS the Z, zero surgery) and braindecode ATCNet (Apache-2.0 upstream). These two show LPC-CMI is not EEGNet-specific.
- HIGH-CAPACITY STRESS-TEST ABLATION: braindecode Deep4Net (~283k params) to show the regularizer suppresses leakage as memorization capacity grows; needs KL warmup + separate lr/weight_decay.
- OPTIONAL GEOMETRIC ABLATION (frozen carrier): pyriemann 0.7 Covariances(estimator='oas') -> TangentSpace(metric='riemann') giving a deterministic well-conditioned Euclidean Z; pair with per-domain recentering. Frozen-feature/post-hoc probe only (not trainable end-to-end).
- DG-FRAMEWORK BASELINES TO IMPLEMENT FIRST (unified harness): ERM; DANN + CDANN (DomainBed classes, MIT — CDANN is the single closest comparison: uniform-target vs our pi_y target); Deep CORAL + MMD + their C-CORAL/C-MMD conditional variants (DomainBed penalty fns, MIT); VREx (most stable risk-based) + GroupDRO (worst-subject); IIB (theoretical foil, MIT) ; EEG-DG Modified-EEGNet (head-to-head MI-EEG DG accuracy/kappa on 2a/2b/OpenBMI); SCLDGN SupConLoss (label-protecting contrastive counterpart).

## Recommended estimator

LPC-CMI realized as a variational posterior-KL UPPER-bound surrogate: L_CMI = E_i KL( q_psi(D | z_i, y_i) || pi_{y_i}(D) ), where q_psi is a CLUBForCategorical-style softmax domain classifier taking [z, one-hot(y)] as input (continuous Z, discrete D), and pi_y(D)=p(D|Y) is the Laplace-smoothed label-conditional prior. Train with the standard CLUB two-step alternation (Step A: fit q_psi by cross-entropy on detached z; Step B: update encoder+task-head with task CE + lambda*L_CMI), which is ALREADY implemented and validated in /home/infres/yinwang/CMI_AAAI/synthetic/sanity_check.py (method 'lpc_prior'). Grounding: Barber-Agakov decomposition I(Z;D|Y)=H(D|Y)-H(D|Z,Y) for discrete D + McAllester-Stratos argument that lower bounds (MINE/InfoNCE) are the WRONG direction for a minimization penalty. REIMPLEMENT the ~25-line CLUBForCategorical class in our own (MIT) code — the Linear95/CLUB repo has NO LICENSE. KEEP AS ABLATIONS/PROBES (all already in sanity_check.py or trivially added): (1) 'marginal' E KL(q(D|Z)||p(D)) — label-erasure failure mode; (2) 'chain' super-label S=(D,Y) — Y-erasure failure mode; (3) 'lpc_uniform' KL to Uniform — the CDANN target, mis-specified under imbalance; (4) CIRCE conditional-HSIC penalty (official MIT PyTorch) as the kernel-side competitor; (5) class-stratified HSIC as cheap kernel baseline. POST-HOC LEAKAGE CERTIFICATION (non-differentiable, offline): permutation HSIC test per Y-stratum; knncmi (Mesner-Shalizi mixed kNN CMI, GPL-3.0, offline-only) as trusted I(Z;D|Y) ground-truth on Protocol-D synthetics; sklearn mutual_info_classif (Ross, BSD, in-stack) stratified per Y as a cheap marginal-MI surrogate.
