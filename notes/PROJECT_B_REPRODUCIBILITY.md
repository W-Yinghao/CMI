# Project B Reproducibility

Three levels: fast contract tests (unit), synthetic evidence regeneration (medium/expensive), and real
EEG bridge / bounded benchmark (data-dependent). Only Level 5.1 is a unit-test gate; 5.2 and 5.3 are
**evidence regeneration**, not unit tests.

## 5.1 Fast contract tests
```bash
python -m h2cmi.router.reasons
python -m h2cmi.router.actions
python -m h2cmi.router.features
python -m h2cmi.router.acar
python -m h2cmi.router.router
python -m h2cmi.eval.router_harness
python -m h2cmi.tests.test_router_smoke
python -m h2cmi.tests.test_real_eeg_bridge
```

## 5.2 Synthetic frozen package regeneration
The frozen Step-2E/2F packages were produced by the SLURM sweep wrappers
(`project_b_step2e_router.slurm` / `project_b_step2f_support.slurm`) across worlds R2/HF3/H_OOD and
seeds; the per-cell python entry points below regenerate one default cell. These runs are longer and
should be treated as **evidence regeneration, not unit tests**.
```bash
python scripts/project_b_eval_router_synthetic.py --out /tmp/project_b_step2e_router
python scripts/project_b_eval_router_support.py --out /tmp/project_b_step2f_support
python scripts/project_b_step2g_report.py \
  --step2e /tmp/project_b_step2e_router \
  --step2f /tmp/project_b_step2f_support \
  --out /tmp/project_b_step2g_report
```

## 5.3 Real EEG bridge / bounded benchmark
```bash
python scripts/project_b_real_eeg_bridge.py \
  --dataset BNCI2014_004 \
  --max_subjects 4 \
  --max_targets 2 \
  --epochs 8 \
  --batch_size 64 \
  --resample 128 \
  --eval_unit subject \
  --support_mode both \
  --max_nested_folds 2 \
  --device cpu \
  --seed 0 \
  --allow_missing_data \
  --out /tmp/project_b_step3a_real_bridge
```
```bash
python scripts/project_b_real_eeg_benchmark.py \
  --datasets BNCI2014_004 \
  --max_subjects 6 \
  --max_targets 4 \
  --epochs 8 \
  --batch_size 64 \
  --resample 128 \
  --eval_units subject,session \
  --support_modes in_source_subject_q95,nested_source_subject_excess_q95 \
  --max_nested_folds 2 \
  --device cpu \
  --seed 0 \
  --allow_missing_data \
  --allow_dataset_failures \
  --out /tmp/project_b_step3c_real_benchmark
```
Note: MOABB data availability and cache state affect runtime. `--allow_missing_data` writes an
availability error record instead of crashing for the bridge smoke, so the pipeline is inspectable even
when a dataset cache is absent. Real runs should be submitted via SLURM (`sbatch`), not on a shared
login node.
