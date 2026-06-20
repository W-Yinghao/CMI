# Code inventory — Tri-CMI experiments

All code I wrote for the experiments. ~1520 LOC across 15 Python modules + 3 slurm scripts.
Excludes cloned reference repos under `repos/` (SCLDGN, SupContrast — not ours).
Everything runs from the `icml` conda env; one runner drives all frameworks via `method:lam[:gamma]` configs.

## Data loaders — `cmi/data/`
| file | LOC | purpose | key API |
|---|---|---|---|
| `moabb_data.py` | 79 | Offline MOABB motor-imagery loader + splits | `load()`, `domain_labels()`, `loso_splits()`, `leave_one_session_splits()`; `DATASET_DEFAULTS` |
| `emotion_data.py` | 92 | SEED (3-class) + DEAP (valence) loaders, windowed | `load("SEED"/"DEAP")` → same `(X,y,meta,classes)` interface |
| `diagnosis_data.py` | 62 | ADFTD SCPS loader (Alzheimer/FTD/Control), `.set` files | `load("ADFTD", binary=)` |
| `cross_dataset.py` | 47 | Protocol C: multi-dataset, 21 common channels aligned | `load_cross()`, `CANON` (channel list) |

## Backbone — `cmi/models/`
| `backbones.py` | 56 | Faithful braindecode encoder + hooked penultimate Z | `HookedBackbone`, `build_backbone()`; EEGNet/ShallowConvNet/Deep4Net/EEGConformer → `(logits, Z)` |

## Methods / frameworks — `cmi/methods/`
| file | LOC | implements (config names) |
|---|---|---|
| `regularizers.py` | 88 | **LPC-CMI core**: `DomainPosteriors`, `empirical_priors`, `kl_to_prior`; methods `erm/marginal/chain/lpc_uniform/lpc_prior` + IIB aux head `iib_ce_h` |
| `contrastive.py` | 43 | domain-aware SupCon: `supcon`, `lpc_supcon` (hybrid); `sup_con_loss()` |
| `dg_penalties.py` | 108 | DomainBed DG: `coral/mmd/irm/vrex/groupdro/dann/cdann`; gradient-reversal + discriminator |

## Training & eval
| file | LOC | purpose |
|---|---|---|
| `cmi/train/trainer.py` | 116 | Two-step alternating trainer (Step A posterior on detached Z; Step B encoder+head), AdamW+cosine, balanced sampler, dispatches all framework families incl. GroupDRO reweight + IIB |
| `cmi/eval/metrics.py` | 66 | balanced acc / macro-F1 / ECE / NLL + **leakage_probe** (frozen-encoder I(Z;D\|Y)) + label_separability |

## Runners — `cmi/`
| file | LOC | protocol(s) |
|---|---|---|
| `run_loso.py` | 179 | Protocol A (LOSO) + B (`--protocol cross_session`) + imbalanced (`--imbalance`) ; auto-dispatches MI/emotion/diagnosis loaders; `--configs method:lam[:gamma]` |
| `run_cross_dataset.py` | 103 | Protocol C (leave-one-dataset-out) |
| `paths.py` | 49 | offline MOABB env config + MI dataset registry |

## Synthetic (Milestone 1) — `synthetic/`
| `sanity_check.py` | 317 | Protocol D DGP + 5 objectives + leakage probe + λ-sweep (produces `results.json`) |
| `make_figure.py` | 35 | Figure 2 (target acc / leakage / label-sep vs λ) |

## SLURM — `scripts/`
| file | LOC | use |
|---|---|---|
| `run_loso.slurm` | 36 | positional-arg LOSO submit: `<dataset> <backbone> "<configs>" <epochs> <tag> <resample>` |
| `run.slurm` | 22 | passthrough wrapper: all flags → `cmi.run_loso` |
| `runmod.slurm` | 22 | generic: `<module> <flags...>` (used for `run_cross_dataset`) |

## Code → experiments map
- **MI LOSO / λ-sweep / frameworks / backbone / cross-session**: `run_loso.py` + `run_loso.slurm`/`run.slurm`
- **Cross-dataset (Protocol C)**: `run_cross_dataset.py` + `runmod.slurm`
- **Imbalanced π_y test**: `run_loso.py --imbalance 0.7`
- **Emotion (SEED/DEAP)**: `run_loso.py --dataset SEED/DEAP` (auto-dispatch)
- **SCPS (ADFTD)**: `run_loso.py --dataset ADFTD`
- **Synthetic (Fig. 2)**: `synthetic/sanity_check.py` + `make_figure.py`

## Review pointers (where correctness matters most)
1. `methods/regularizers.py` — the LPC-CMI estimator + ablations + priors (the core claim).
2. `train/trainer.py` — the two-step alternation, λ warm-up, and per-framework loss dispatch.
3. `methods/dg_penalties.py` — DomainBed baselines (CORAL/MMD/IRM/VREx/GroupDRO/DANN/CDANN) — first-pass per-framework λ, reimplemented (verify against DomainBed).
4. `eval/metrics.py::leakage_probe` — the headline conditional-leakage metric.
5. `run_loso.py::_imbalance_subsample` — the imbalance induction.
6. Data loaders — windowing, normalization (per-trial z-score), label maps, channel alignment.
