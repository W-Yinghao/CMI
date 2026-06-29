# CIGL Phase 2-real — Exploratory Source-Only Probes

Phase 2-real takes the Phase-2 audit machinery (`docs/CIGL_09_PHASE2_LEAKAGE_AUDIT.md`) to **real
EEG**, still as a **diagnostic** — it trains **no** CIGL regularizer, sweeps **no** λ, makes **no**
SOTA claim. It answers, on real data:

> Does a source-only **GraphCMINet-ERM** encoder produce learned graph objects that carry
> label-conditional **source-domain (subject)** leakage — `I(Z_g;D|Y)`, `(1/C)Σ_v I(Z_v;D|Y)`,
> `I(A;D|Y)` — above a retrained within-label permutation null, with **seed-stable** maps?

Runner: `scripts/run_cigl_phase2_real_probe.py`. Modules: `cmi/eval/probe_splits.py`,
`cmi/eval/graph_map_stability.py`, and the `train_idx`/`val_idx` extension of
`cmi/eval/graph_leakage.py:audit_graph_objects`.

> **Exploratory, not benchmark evidence.** Every output file carries `meta.exploratory = true`. These
> numbers inform the Gate-2 decision; they are not a results table and must never be presented as one.

---

## 1. Strict source-only rule (unchanged from Phase 2)

- The held-out **target subject is never used** — not in encoder training, feature extraction, or the
  probe. The audit measures leakage **among source subjects** (`D = source subject id`).
- **No target labels, no target covariates** anywhere. Recorded as
  `meta.used_target_labels = false`, `meta.used_target_covariates = false`,
  `meta.setting = "strict_source_only_DG"`.
- Three disjoint source subsets per fold/seed: **encoder-train** (trains GraphCMINet-ERM),
  **probe-train** (fits `q(D|object,Y)`), **probe-val** (held-out KL). The encoder never sees the
  probe-val trials.

## 2. Probe train/val split must be support-aware by (Y, D)

`D = subject_id`, so a plain random split can leave a subject entirely in probe-val — the classifier
is then asked to predict an **unseen** domain, which is meaningless. `probe_splits.py` splits at the
**trial level**, and within each `(Y, D)` cell with `>= min_per_cell` trials places some in train and
some in val so **every subject appears in both** splits. Cells too small to split are kept in
**train** and reported as **low-support** (`n_cells_low_support`, `low_support_cells`,
`missing_val_domains`) — never silently dropped to manufacture a clean number. Node features are split
at the trial level (node-rows of a trial never straddle the boundary).

## 3. Permutation null (retrained, train-split only)

Identical to Phase 2: the null permutes `D` **within label, over the probe-train indices only**, and
**retrains** the probe; validation `D` is untouched and the train-split `π_y(D)` is preserved.
`audit_graph_objects(..., train_idx=, val_idx=)` threads the support-aware split through and uses
`permute_idx = train_idx`.

**`n_perm >= 50`** for real probing (preferably 99). `n_perm` of 5–10 is an engineering/dry-run check
only; its smallest attainable p is `1/(n_perm+1)`.

## 4. Seed stability of the maps

A node/edge leakage map is credible only if it points at the **same** electrodes/edges across seeds.
`graph_map_stability.py` reports the mean pairwise rank correlation across per-seed maps and compares
it to a **random-map null** (each map's entries independently shuffled). A map no more consistent than
the null is noise and must not become a "subject-fingerprint" figure.

## 5. Output schema (`results/cigl/phase2_real/`)

Per `(fold, seed)`: `<fold>_seed<S>.json` with `meta` (exploratory flags, dataset, fold, seed,
n_perm, epochs, n_classes, n_domains, commit_hash, config_hash, heldout_subject), `source_info`
(source subjects, enc/probe-pool sizes, encoder-split diagnostics), `probe_split_diagnostics` (the
support-aware split report), and `graph`/`node`/`edge` blocks (kl_mean, permutation_{mean,std,p},
domain_acc, prior_acc, leakage_advantage, kl_ci; node adds `node_leakage_map` + `_path`; edge adds
`edge_leakage_top_k` + `edge_leakage_map_path`). Per fold: `<fold>_summary.json` with `meta`
(incl. `gate_alpha`), `per_seed` rows whose graph/node/edge blocks carry `kl_mean`,
`permutation_{mean,p}`, **`positive_excess`**, **`clears_null`**, **`gate_alpha`**, and `map_stability`
(node/edge stability + random null). Map `.npy` sidecars and the run `.json` are gitignored generated
artifacts; the JSON carries compact inline summaries.

## 6. Gate-2 (real) decision

### 6.1 Two distinct verdicts (do not conflate)

- **positive excess** `:= kl_mean > permutation_mean`. A **directional signal only**. It is **NOT**
  sufficient for a Gate-2 pass — a weak, non-significant difference can have positive excess.
- **clears null** `:= kl_mean > permutation_mean AND permutation_p <= gate_alpha`
  (default `gate_alpha = 0.05`). This is the **binding** Gate-2 significance criterion.

`permutation_p` is the `(+1)`-smoothed within-train permutation p-value; its floor is `1/(n_perm+1)`,
so **real** runs need `n_perm >= 50` (preferably 99). A dry-run (`n_perm` of 5–10) typically yields
`clears_null = false` for every object — that is correct and intended; the dry-run validates the
pipeline, not significance.

### 6.2 Decision (read the per-seed `clears_null` table + map-stability across `>= 3` seeds)

- **A — proceed to full CIGL (Phase 3):** at least **two of three** objects (graph/node/edge) have
  `clears_null = true` in **>= 2/3 seeds**, **and** the node or edge map is `above_random` stable.
- **B — narrow to Node-CMI:** only **node** `clears_null` robustly (and node map `above_random`).
- **C — narrow to Edge-CMI:** only **edge** `clears_null` robustly (and edge map `above_random`).
- **D — pivot to diagnostic-only:** no object `clears_null` robustly → do not claim a regularizer;
  reframe CIGL as an audit/diagnostic framework.

The runner prints **both** counts (`positive_excess` and `clears_null` per seed) and writes
`positive_excess` / `clears_null` / `gate_alpha` into every per-seed object block, but makes **no**
decision — the reviewer decides A/B/C/D from the artifacts using the `clears_null` criterion. No
Phase 3 / λ-sweep is authorized until a real Gate-2 pass on the significance criterion.

## 7. Compute note

GraphCMINet-ERM training on real EEG (e.g. BNCI2014_001) is heavy on CPU. The login node has no GPU;
the binding acceptance here is the **synthetic dry-run**. The real evidence run (`--seeds 0 1 2
--n_perm 50`) should be executed on GPU via sbatch. The runner builds `GraphCMINet` directly (no
braindecode) and loads data offline via the MOABB datalake cache (`cmi/paths.py`); if the cache is
absent it **fails clearly** and never downloads.

## 8. Acceptance

```bash
pytest -q tests/test_graphcmi_backbone.py tests/test_graph_regularizers.py tests/test_graph_leakage.py \
         tests/test_probe_splits.py tests/test_graph_map_stability.py
python scripts/smoke_graphcmi.py --device cpu
python scripts/smoke_graph_leakage.py --device cpu
python scripts/run_cigl_phase2_real_probe.py --dry_run_synthetic --device cpu --seeds 0 1 2 --n_perm 10
# if the offline EEG cache is present (heavy on CPU; prefer GPU/sbatch):
python scripts/run_cigl_phase2_real_probe.py --dataset BNCI2014_001 --device cpu --seeds 0 1 2 --n_perm 50 --max_folds 1
```
