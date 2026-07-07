# Dataset selection — grounded in what the baselines used

We anchor on the datasets the most-related MI-EEG DG baselines report, so our numbers
are **directly comparable**. All are already cached offline in the datalake (no download).

## What the related papers used
| Paper (role) | Datasets used |
|---|---|
| **EEG-DG** (JBHI'24, most-related app baseline) | **BCI-IV-2a, BCI-IV-2b** (+ simulative). Reports **2a: 81.79%**, **2b: 87.12%** — ⚠️ but under a **within-subject cross-session** protocol (sessions/splits of the *same* subjects as domains; target subject **seen**), NOT subject-independent LOSO. Their numbers are **not directly comparable** to our cross-subject LOSO; reimplement their loss on our protocol for a fair head-to-head. |
| **SCLDGN** (TBME'24/25, contrastive DG) | 6 datasets via parsers: **2a, 2b, BCI-IV-3a, GIST(=Cho2017), HGD(=Schirrmeister), Korea(=Lee2019/OpenBMI), PhysionetMI**. |
| **TSMNet/SPDDSMBN** (NeurIPS'22, geometric DG) | MOABB: **BNCI2014001(=2a), BNCI2015001, Lee2019, Lehner2020, Stieger2021, Hehenberger2021** + a workload set. inter-session & inter-subject TL. |
| Survey Table 1 | PhysionetMI, 2a, 2b, GigaScience(Cho2017), OpenBMI(Lee2019), Stieger2021, Yi2014, BCI-III-IVa … |

**Consensus core: 2a and 2b appear in EEG-DG + SCLDGN + (2a in) TSMNet** → universal MI-DG benchmark.

## Our selection (all in `/projects/EEG-foundation-model/datalake/raw`, MOABB-loadable offline)
| Tier | MOABB id | folder | subj | cls | sess | role |
|---|---|---|---|---|---|---|
| **Core** | BNCI2014_001 (2a) | MNE-bnci-data/…/001-2014 | 9 | 4 | 2 | LOSO + cross-session; **EEG-DG/TSMNet/SCLDGN match** |
| **Core** | BNCI2014_004 (2b) | MNE-bnci-data/…/004-2014 | 9 | 2 | 5 | LOSO + cross-session; **EEG-DG/SCLDGN match** |
| **Scale** | Lee2019_MI (OpenBMI) | MNE-lee2019-mi-data | 54 | 2 | 2 | large-scale LOSO; **TSMNet/SCLDGN match** |
| Breadth | Cho2017 (GIST) | MNE-gigadb-data | 52 | 2 | 1 | cross-dataset binary; **SCLDGN match** |
| Breadth | Schirrmeister2017 (HGD) | MNE-schirrmeister2017-data | 14 | 4 | 1 | strong-encoder; **SCLDGN/TSMNet ref** |
| Optional | BNCI2015_001 | MNE-bnci-data/…/001-2015 | 12 | 2 | 2-3 | TSMNet match (present, confirmed) |
| Optional | Stieger2021 | MNE-Stieger2021-data | 62 | 2/4 | 7-11 | cross-session at scale; TSMNet match |

## π_y showcase (key: pi_y correction needs label-domain imbalance — balanced MOABB MI hides it)
Balanced MI → p(D|Y)≈p(D)≈uniform → lpc_prior≈lpc_uniform≈marginal (confirmed in v2). To show the
*specific* pi_y contribution on real data:
- **Imbalanced-LOSO** (`cmi.run_loso --imbalance 0.7`): per source domain, keep all of its "preferred"
  class (g mod n_cls) but only (1-ρ) of others → p(D|Y) far from uniform (verified: .33→.44/.13/.44).
  Target stays balanced. Direct real-EEG analogue of the synthetic; tests lpc_prior > uniform/marginal/CDANN.
- **Protocol C cross-dataset** (`cmi.run_cross_dataset`): 2a/Lee2019/Cho2017, **21 common channels**,
  leave-one-dataset-out. **Domain D = dataset:subject** (subject-level, as in code `run_cross_dataset.py`),
  so the regularizer suppresses subject leakage within the source datasets while testing transfer to an
  unseen dataset/montage. (If we instead want to claim pure *device/dataset*-level invariance, switch
  D=dataset — fewer, denser domains; would change I(Z;D|Y) semantics. Decide per paper claim.)

## Protocol → domain D mapping
- **A — within-dataset LOSO** (2a, 2b first): D = subject. Headline result table.
- **B — cross-session** (2a 2-sess, 2b 5-sess; Lee2019 2-sess): D = subject×session; test unseen session.
- **C — cross-dataset binary MI** (2b ↔ Lee2019 ↔ Cho2017, left vs right): D = dataset; channel intersection / standard montage.

## Emotion-recognition extension (cross-task generality)
Emotion recognition is the *most-studied* cross-subject DG task in the survey (13 papers vs MI's 11),
so an emotion result counters the "MI-only / too-applied" reviewer risk and shows Tri-CMI is
task-agnostic. Loader `cmi/data/emotion_data.py` (same `(X,y,meta,classes)` interface):
| dataset | path | subj | ch/fs | classes | windows |
|---|---|---|---|---|---|
| **SEED** | `/projects/EEG-foundation-model/SEED/*.mat` | 15×3sess | 62 @200Hz | 3 (neg/neu/pos) | 4s, ≤20/trial |
| **DEAP** | `.../DEAP/data_preprocessed_python/sXX.dat` | 32 | 32 @128Hz | 2 (valence@5) | 4s, drop 3s baseline |
Domain D = subject (LOSO). Run via the same `cmi.run_loso` (auto-dispatches by dataset name).

## Build order
1. **2a + 2b LOSO** (9 subjects each — fastest complete result; direct EEG-DG comparison).
2. + cross-session on 2a/2b.
3. + Lee2019 LOSO (scale, 54 subj — heavier, V100/A100).
4. (optional) Cho2017 / HGD / cross-dataset.
