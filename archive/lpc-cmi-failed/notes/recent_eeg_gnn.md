# Recent EEG-GNN Landscape (2023-2025) — Positioning Report for GraphCMINet

## Scope and verification stance
GraphCMINet is a **raw-signal EEG-GNN** for **cross-subject domain generalization (DG, LOSO)** that minimizes **label-conditional** domain leakage `I(·;D|Y)` at **three granularities** — graph `I(graph_Z;D|Y)`, node `Σ_v I(Z_v;D|Y)` (which electrodes leak subject identity), edge `I(A;D|Y)` (the per-sample learned adjacency is a subject fingerprint) — **non-adversarially**, on a **named domain D = subject**. Below, every paper is tagged on the four load-bearing axes (is-EEG / input DE-vs-raw / protocol within-vs-cross-subject / domain-method level+adversarial?+conditional-on-Y?), with DE-vs-raw mismatches flagged for fair comparison.

Two hard filters applied throughout:
- **Confirmed-EEG only** as a baseline. BrainIB is fMRI (graph-IB scaffold only, not an EEG DG baseline). GraphS4mer/DSTGNN/SDN-Net are EEG but on seizure/auditory tasks (different benchmark).
- **DE vs raw**: GraphCMINet forbids precomputed DE/PSD and uses a raw PowerLayer node encoder. **Almost the entire SEED/SEED-IV GNN line (DGCNN, RGNN, SOGNN, ST-SCGNN, V-IAG, GDDN, FreqDGT, MoGE) is DE/PSD-input**, so those are *method-level / accuracy-axis* comparators, not same-input baselines. The only same-input (raw) graph precedent is **LGGNet** (within-subject only) and, on other tasks, GraphS4mer/DSTGNN.

---

## 1. Landscape by method family

### Family A — Learnable per-sample adjacency, NO domain term (the "self-constructing graph" line)
These establish GraphCMINet's *backbone* (per-sample learned `A(x)`) but do **zero** subject/domain invariance and nothing conditional on Y. They are fair cross-subject baselines and the cleanest "architecture without leakage control" foils.

- **SOGNN** (Front. Neurosci. 2021, PMC8221183) — EEG, **DE**, **cross-subject LOSO**. Self-organized per-sample `A = softmax(tanh(VW)tanh(VW)ᵀ)`, top-k=10. **No** domain invariance. SEED **86.81%**, SEED-IV **75.27%** (LOSO). The canonical learnable-per-sample-adjacency precursor; its learned A *can* encode subject identity → directly motivates our edge term.
- **ST-SCGNN** (JBHI 2023, doc 10329957, PMID 38015677) — EEG, **DE + functional-connectivity features**, **cross-subject** (+ DOC clinical generalization split). Self-constructing per-sample adjacency. **No** domain/leakage objective at any granularity. SEED **85.90%**, SEED-IV **76.37%**.
- **V-IAG** (TAFFC 2021, doc 9373917) — EEG, **DE**, protocol mixed (within-subject SEED 96.43% is inflated; do not cite as DG). Instance-adaptive **+ variational/probabilistic** adjacency (distribution over A). Important precedent for *per-sample + probabilistic* graph, but models measurement **uncertainty**, NOT subject leakage, and is **not** label-conditional. Use to sharpen the contrast: variational graph ≠ variational IB on D.
- **GraphS4mer** (arXiv:2211.11176) — EEG (seizure, among 3 biosignals), **raw**, **cross-subject/patient-wise** (TUSZ patient-disjoint). Self-attention dynamic per-sample GSL + smoothness/degree/sparsity regularization. **No** subject/domain invariance, no conditional term. TUSZ AUROC **0.906**. A **raw-input, cross-subject, learnable-adjacency** EEG-GNN baseline and a borrowable graph-construction precedent — but **different task** (seizure, not emotion/MI) and **no** I(·;D|Y) overlap.
- **DSTGNN** (Meas. Sci. Technol. 2025, DOI 10.1088/1361-6501/adfb95) — EEG (seizure), **raw**, **patient-independent**. Effective-connectivity (ADTF directed) dynamic adjacency. **No** domain invariance; patient-independence is split-based only. Weak baseline / borrow-idea (ADTF directed connectivity, channel-count robustness). Verify preprint-vs-published numbers before quoting.

### Family B — Domain handling at GRAPH/pooled level only (no node, no edge)
These DO attack subject variability, but only on the pooled graph embedding — they cannot express the node- or edge-level leakage GraphCMINet targets.

- **FreqDGT** (arXiv:2506.22807, MIND 2025) — EEG, **rPSD band-power features**, **cross-subject LOSO**. **ADVERSARIAL** subject-disentanglement: subject-encoder vs emotion-encoder + subject discriminator in a minimax game, on the **graph/pooled** embedding. **NOT conditional on Y** (marginal I(Z;D)). Adaptive Dynamic Graph Learning gives per-sample adjacency, but **the adjacency itself is not regularized against subject leakage**. LOSO SEED **81.1%** (F1 81.9), SEED-IV **71.9%** (F1 72.6, 4-class), FACED **62.3%** (SEED/FACED run **binary**). **The closest named competitor** in this design space.
- **PR-PL** (TAFFC 2024, arXiv:2202.06509) — EEG, **DE**, **cross-subject LOSO** but **transductive DA** (sees unlabeled target). Adversarial DANN-style source-vs-target alignment + prototypes; **marginal**, not label-conditional; **not a GNN** (MLP 310-64-64-64). SEED 85.56% / SEED-IV 74.92% (cross-subject cross-session LOSO); single-session 93.06%. **DA, not pure DG** — flag the DA-vs-DG mismatch when comparing.
- **MI-EEG** (Expert Syst. Appl. 2024, DOI 10.1016/j.eswa.2023.122777) — EEG, **DE**, **cross-subject**. **Non-adversarial MI** disentanglement: MI-min between latent splits + MI-max(invariant features; Y). **Methodologically the most similar** ("MI not adversarial"), but **graph/global level only**, **not a GNN**, and the MI-min is split-vs-split, **not an explicit label-conditional I(Z;D|Y)**. SEED **85.4%** (86.7% after unlabeled calibration). Cite as the direct "MI-not-adversarial" predecessor.

### Family C — Domain handling at TWO graph granularities (edge + representation), but by decomposition not conditional-MI
- **GDDN** (TAFFC 2024, **correct DOI 10.1109/TAFFC.2024.3371540**, doc 10453943, ChenCZ24) — EEG, **STFT/DE-family features**, **cross-subject / cross-individual + cross-dataset**. **Non-adversarial DISENTANGLEMENT** of (1) graph **connectivity/adjacency** (edge) **and** (2) graph **representation** into common vs subject-specific, + domain-adaptive classifier aggregation. **This is the nearest edge-level prior**: it already treats the adjacency as a subject carrier. BUT: common/specific **decomposition, not conditional-MI**; **not** explicitly label-conditional I(·;D|Y); **no per-node electrode-wise leakage term**; DE-family input. Exact numbers paywalled (an unverified ~92.54% SEED snippet — do not cite). The **must-cite nearest edge-level prior**.

### Family D — Node-level domain invariance (adversarial, unconditional) — the direct node-term prior
- **RGNN** (TAFFC 2020, arXiv:1907.07835) — EEG, **DE**, protocol both (subject-independent SEED ~85.30%, SEED-IV ~73.84% — the **strongest honest-LOSO small-param GNN**). **NodeDAT** = per-node gradient-reversal domain-adversarial training (per-electrode DANN). **The key prior for node-level invariance**, but **adversarial**, **unconditional** (marginal per node), and on a **FIXED biologically-initialized distance adjacency** (not per-sample learnable). GraphCMINet's `Σ_v I(Z_v;D|Y)` is the **non-adversarial, label-conditional generalization of NodeDAT** — and produces a per-channel leakage map NodeDAT's symmetric discriminator cannot.
- **DGCNN** (TAFFC 2018, DOI 10.1109/TAFFC.2018.2817622) — EEG, **DE**. Single **global static learnable** A + ChebNet. **No** domain term. LOSO SEED **79.95%**, SEED-IV **52.82%** (within-subject 90.4% — **never** cite as DG). Backbone-ancestor baseline.

### Family E — Architectural DG (no invariance loss at all)
- **MoGE** (IEEE BIBM 2024, doc 10822354) — EEG, **DE**, **cross-subject DG**. **Plain ERM**, no adversarial/IB/MI/disentanglement (confirmed: only Linear head + CE). DG comes purely from a **Sparse Mixture-of-Graph-Experts** that routes each electrode to a specialized expert (node-level **routing**, but an inductive-bias decomposition, **not** a leakage penalty, and **not** conditioned on Y or D). Adjacency is a single **global** learnable parameter, **not per-sample**. **Strong DG baseline** (SEED **88.0%**, SEED-IV **74.3%**, SEED-V **81.8%**) and an **orthogonal** architectural-DG alternative to cite — zero overlap with our conditional-MI contribution.

### Family F — Raw-signal graph templates (satisfy our input policy) but within-subject / no domain term
- **LGGNet** (TNNLS 2023, arXiv:2105.02786) — EEG, **RAW**, **within-subject only**. Multi-scale temporal convs + **PowerLayer = log(avgpool(x²))** (the raw band-power surrogate GraphCMINet adopts) + local region graphs + a global trainable adjacency. **No** subject/domain objective. **The same-input (raw) graph baseline** and the encoder template — its cross-subject behavior is *open*, which is precisely GraphCMINet's opening.

### Family G — Non-GNN raw-signal cross-subject DG (method-level comparators)
- **SCLDGN** (TBME 2025, **correct DOI 10.1109/TBME.2024.3432934**, PMID 39046861) — EEG **motor imagery**, **raw**, **cross-subject DG (LOSO, calibration-free)**. **Non-adversarial**, **class-conditional**: deep CORAL + supervised-contrastive with domain-agnostic mixup. **Shares our non-adversarial + class-conditional + pure-DG stance**, but: **CNN (EEGNet-style), not a GNN** — no nodes/edges/adjacency; granularity = **pooled embedding only**; CORAL/SupCon distribution-alignment, **not** explicit conditional-MI. Strong non-adversarial cross-subject DG method-baseline (MI task; method-level vs SEED).
- **CLISA** (TAFFC 2022, arXiv:2109.09559) — EEG, **raw**, **cross-subject**. **Non-adversarial** contrastive inter-subject alignment; **stimulus/label-conditional** (positives = same-stimulus across subjects). **Global embedding only**, **not a GNN**, no MI-min. SEED ~86.3%. The canonical non-adversarial **label-conditional** subject-invariance method on raw EEG — but global, not graph; cite for the conditional-alignment framing.
- **MI-DG (knowledge-distillation + CORAL)** (Bioengineering 2025, DOI 10.3390/bioengineering12050495) — EEG MI, **DE/multi-band features**, **cross-subject LOSO**. Non-adversarial (distillation + CORAL) but **global/marginal, not conditional**, **not a GNN**. BCIC-IV-2a 60.07%, KU 81.80% (LOSO). MI accuracy baseline only; **not** an EEG-GNN.
- **CoMET** (arXiv:2509.00314) — EEG, **raw**, **cross-subject LOSO** SSL foundation model. **No** domain invariance (plain instance-discrimination, subject-agnostic), **not a GNN**. Positioning contrast: a 2025 raw-EEG model that transfers cross-subject **without** explicit domain invariance. Not a direct baseline (no SEED/DEAP cross-subject numbers; beats CBraMod).

### Family H — Subject-disentanglement, but inverse problem or regression (borrow-idea only)
- **Brainprint/ADN** (TIFS 2025, doc 11138010) — EEG **biometric ID** (subject = TARGET to KEEP). **Inverse** of DG; adversarial disentanglement, **non-graph**. Not a baseline; borrow the decorrelation/disentanglement machinery and contrast direction-of-removal.
- **SDN-Net** (arXiv:2501.08693) — EEG **speech-envelope regression**. **Non-adversarial MI-min** (vCLUB) of I(envelope; subject), **global, UNCONDITIONAL I(Z;D)**, non-graph, regression. Cite to show MI-disentanglement in EEG exists but is **non-graph, non-conditional, regression-only**.

---

## 2. Trends (2023-2025)
1. **Graph construction has converged on per-sample / dynamic learnable adjacency** (SOGNN → ST-SCGNN → FreqDGT/ADGL → GraphS4mer), displacing fixed distance/PLV graphs. **The field has not noticed that this per-sample A is itself the strongest subject fingerprint** — nobody regularizes A against subject leakage. This is GraphCMINet's cleanest open lane (edge term).
2. **DG vs DA discipline is tightening**: honest LOSO is now standard for the better emotion-GNN papers (SOGNN, ST-SCGNN, MoGE, FreqDGT, GDDN). But several headline numbers are still inflated or non-comparable — DGCNN/LGGNet within-subject, PR-PL transductive DA, FreqDGT binary SEED. Protocol hygiene is a differentiator.
3. **Domain handling, when present, is overwhelmingly (a) adversarial and (b) marginal and (c) graph/pooled-level**: FreqDGT/PR-PL (adversarial, marginal, pooled), RGNN-NodeDAT (adversarial, unconditional, node). **Non-adversarial MI** appears only off-graph (MI-EEG, SCLDGN, CLISA, SDN-Net) and **label-conditional** appears only globally (CLISA, SCLDGN class-conditional, MI-EEG partial).
4. **Raw-signal graphs remain rare** and are confined to seizure (GraphS4mer, DSTGNN) or within-subject (LGGNet). On SEED/SEED-IV the entire GNN line is DE/PSD. **Raw-signal cross-subject EEG-GNN for emotion is essentially unoccupied.**

---

## 3. SOTA snapshot (cross-subject / LOSO)
- **SEED 3-class LOSO**: MoGE **88.0%** (DE, ERM-MoE) > SOGNN 86.81% ≈ CLISA 86.3% > ST-SCGNN 85.90% ≈ RGNN ~85.30% ≈ MI-EEG 85.4% > FreqDGT 81.1% (binary, DG-adversarial) > DGCNN 79.95%.
- **SEED-IV 4-class LOSO**: ST-SCGNN **76.37%** ≈ SOGNN 75.27% > MoGE 74.3% ≈ PR-PL 74.92% (DA) ≈ RGNN ~73.84% > FreqDGT 71.9%.
- **Honest GNN targets to beat (DE):** RGNN 85.30 / 73.84; **DG-GNN comparator:** FreqDGT 81.1 / 71.9.
- **Cross-subject MI DG (raw, method-level):** SCLDGN (multi-MI/ME datasets, SOTA-claimed); BCIC-IV-2a LOSO ~60% / KU ~82% (CORAL+distillation baseline).
- **Caveat for GraphCMINet:** these top numbers ride on **DE features we forbid**; our raw stem may trail RGNN's 85.30 before CMI gains. **Headline the generalization-gap (worst-subject, leakage_kl) and the node/edge leakage maps, not absolute SEED accuracy** — and report a fair raw-graph baseline (graphcmi λ=0) to isolate the CMI delta.

---

## 4. Fair-baseline picks for the GraphCMINet paper
- **Same-input (raw) graph:** LGGNet (re-run cross-subject — its DG behavior is open) + our own raw-graph baseline (graphcmi λ=λ_node=λ_edge=0).
- **DG-GNN comparator (DE, accuracy axis):** FreqDGT (closest competitor), GDDN (nearest edge-level prior), MoGE (architectural DG). Flag DE-vs-raw input mismatch.
- **Node-term prior (must-cite + re-implement):** RGNN-NodeDAT (~10 LOC GRL + per-node head) as the adversarial, unconditional comparator to `Σ_v I(Z_v;D|Y)`.
- **Honest LOSO GNN accuracy anchors:** RGNN, SOGNN, ST-SCGNN, DGCNN.
- **Non-adversarial MI / conditional foils (off-graph):** MI-EEG, SCLDGN, CLISA (cite to show MI/conditional invariance exists but never at node+edge graph granularity).
- **Estimator-robustness foils:** cHSIC (kernel CMI), matrix-Rényi MI (BrainIB estimator) to show the leakage story is estimator-independent.
- **Do NOT use as fair EEG-GNN baselines:** BrainIB (fMRI), PR-PL (DA + non-GNN), M-GCN (proprietary, feature-domain not subject-domain), brainprint/ADN (inverse problem), SDN-Net (regression), CoMET (SSL non-GNN). GraphS4mer/DSTGNN are raw EEG-GNN but seizure-task — cite as raw cross-subject EEG-GNN precedents, not same-benchmark.

---
## Comparison vs GraphCMINet

| Method (year) | Input | Graph | Protocol | Domain-method: level / adversarial? / conditional-on-Y? | GAP GraphCMINet fills |
|---|---|---|---|---|---|
| **GraphCMINet (ours)** | **RAW** (PowerLayer) | **per-sample learnable A(x)** | cross-subject LOSO (pure DG) | **node + edge + graph** / **NON-adversarial (CMI)** / **YES, label-conditional I(·;D|Y) + π_y** | — |
| FreqDGT (2025) | DE/rPSD | per-sample (ADGL) | LOSO (SEED/FACED binary) | graph/pooled / **adversarial** / **no (marginal)** | non-adversarial; conditional; +node `Σ_v I(Z_v;D|Y)`; +edge `I(A;D|Y)` (A unregularized in FreqDGT); raw input |
| GDDN (2024) | DE/STFT | learned connectivity (disentangled) | cross-subject + cross-dataset | edge + graph / non-adversarial (decomposition) / **no (not explicit)** | conditional-MI vs common/specific decomposition; **per-node electrode-leakage term**; raw input |
| RGNN-NodeDAT (2020) | DE | **fixed** distance A | subject-independent | **node** / **adversarial (GRL)** / **no (unconditional)** | non-adversarial; **conditional**; +edge+graph; per-sample A; raw input; node leakage MAP |
| MoGE (2024) | DE | global learnable A (MoE experts) | cross-subject DG | node-routing / **none (ERM)** / no | an explicit conditional-MI leakage penalty (vs inductive-bias routing); per-sample A; edge term; raw |
| SOGNN (2021) | DE | per-sample `softmax(GGᵀ)`+top-k | LOSO | **none** / — / — | all leakage control (none present); raw input |
| ST-SCGNN (2023) | DE+FC | per-sample self-constructing | cross-subject | **none** / — / — | all leakage control (none); raw input |
| V-IAG (2021) | DE | per-sample + variational A | mixed | none (models uncertainty) / — / no | variational IB on **D** (not measurement noise); conditional; node+edge |
| MI-EEG (2024) | DE | none (not a GNN) | cross-subject | graph/global / non-adversarial (MI) / **partial** | node + edge graph granularity; explicit I(Z;D|Y); per-sample A |
| SCLDGN (2025) | RAW | none (CNN) | LOSO (MI) pure DG | pooled / non-adversarial (CORAL+SupCon) / **yes (class-cond.)** | GNN node/edge/adjacency leakage; explicit conditional-MI (vs distribution alignment) |
| CLISA (2022) | RAW | none (CNN) | cross-subject | global embedding / non-adversarial (contrastive) / **yes (stimulus-cond.)** | node/edge/graph granularity; learned-adjacency leakage; MI objective |
| LGGNet (2023) | RAW | local region + global A | **within-subject only** | **none** / — / — | cross-subject DG; all leakage control; per-sample A |
| DGCNN (2018) | DE | **global static** learnable A | LOSO (don't cite 90.4 within) | none / — / — | per-sample A; all leakage control; raw |
| GraphS4mer (2022) | RAW | per-sample self-attn A | cross-patient (seizure) | **none** / — / — | subject/domain invariance (none); conditional-MI at 3 granularities |

---
## Novelty gap

HONEST VERDICT: GraphCMINet's core claim — a NON-ADVERSARIAL, LABEL-CONDITIONAL mutual-information penalty I(·;D|Y) on a NAMED subject domain D, imposed simultaneously at BOTH per-node features Σ_v I(Z_v;D|Y) AND the per-sample learned adjacency I(A;D|Y) — is GENUINELY UNCLAIMED in the EEG-GNN literature as of 2025. No recent EEG-GNN does label-conditional node+edge domain-MI.

The claim survives adversarial decomposition into its parts, each bracketed by a near-miss that does NOT cover it:
- NODE level: only RGNN-NodeDAT does per-node domain invariance, but it is ADVERSARIAL (gradient-reversal), UNCONDITIONAL (marginal per node), and on a FIXED distance graph. Our node term is the non-adversarial, label-conditional generalization of NodeDAT, and uniquely yields a per-channel "which electrodes leak subject identity" map (a symmetric adversarial discriminator cannot produce one). UNCLAIMED as conditional/non-adversarial.
- EDGE level: GENUINELY NEW and the strongest novelty. The whole field has converged on per-sample learnable adjacency (SOGNN/ST-SCGNN/FreqDGT/GraphS4mer) yet NOBODY regularizes A against subject leakage, even though the fingerprinting literature shows the learned adjacency is the strongest subject discriminator. GDDN is the ONLY near-miss: it disentangles connectivity into common/subject-specific — but by feature DECOMPOSITION, not conditional-MI, not explicitly conditioned on Y, on DE/STFT input. So "I(A;D|Y) on a per-sample learned adjacency" is unclaimed; we MUST cite GDDN as the nearest edge-level prior and clearly differentiate (decomposition vs conditional-MI; no per-node term; DE vs raw).
- GRAPH/global level: this is the LEAST novel part — non-adversarial conditional invariance on a pooled embedding exists off-graph (MI-EEG: MI-min, partial-conditional; SCLDGN: CORAL+SupCon, class-conditional; CLISA: contrastive, stimulus-conditional). So claim novelty ONLY for the combination (graph term is our existing lpc_prior, not the contribution).
- RAW-SIGNAL graph: raw EEG-GNN exists (LGGNet within-subject; GraphS4mer/DSTGNN on seizure), so "raw EEG-GNN" alone is NOT novel. Novel is raw-signal + cross-subject-DG + emotion, which is essentially unoccupied (LGGNet never ran cross-subject).

WHAT WE MUST CITE/COMPARE TO BE SAFE: (1) RGNN-NodeDAT — must-cite + re-implement as the adversarial node-term foil. (2) GDDN — must-cite as the nearest edge-level (adjacency-as-fingerprint) prior; differentiate on conditional-MI-vs-decomposition + per-node term + raw input. (3) FreqDGT — the closest named competitor (per-sample adjacency + subject-invariance); differentiate on non-adversarial + conditional + node/edge (its A is unregularized). (4) MI-EEG + SCLDGN + CLISA — non-adversarial / conditional invariance predecessors (off-graph); show they never reach node+edge graph granularity. (5) MoGE — orthogonal architectural-DG baseline. (6) LGGNet — same-input raw baseline (run cross-subject). (7) BrainIB/GIB — graph-IB scaffold, but BrainIB is fMRI and task-(not-domain)-conditional; cite as method scaffold only, not an EEG baseline.

THREE HONESTY CAVEATS that protect the claim: (a) frame the graph-level term as prior (our lpc_prior), reserve novelty for node+edge+conditional; (b) report a fair raw-graph baseline (graphcmi λ=0) and headline generalization-gap/leakage maps, since raw input may trail DE-based RGNN 85.30 on raw accuracy; (c) flag DE-vs-raw input mismatch on every DE baseline (RGNN/FreqDGT/GDDN/MoGE/SOGNN/ST-SCGNN) so accuracy comparisons are method-level, not apples-to-apples.