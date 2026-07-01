"""ACAR V5 Stage-1B REAL substrate trainer — the VIEW BOUNDARY. Constructed by its factory with a gate-issued
Stage1BExecutionContext, AFTER the gate. NO heavy import at module level (torch is lazy inside the numeric backend, imported by
real_eegnet_trainer). It reads signal ONLY via dataset_view.read_windows and labels ONLY via dataset_view.read_label (the FIT
training view) — never raw cohort roots, never the reader directly, never a filesystem scan — then hands the ALREADY-READ FIT data
to the numeric core (real_eegnet_trainer) which fits under training_config and emits the model + config files into the per-ref
output dir. It does NOT emit feat_dump (that is the separate label-free embedding dumper's job).
"""
from __future__ import annotations
from acar.v5.substrate import stage1b_output_layout as LO


class RealTrainerError(RuntimeError):
    pass


class RealSubstrateTrainer:
    def __init__(self, context, backend=None):
        if context is None:
            raise RealTrainerError("RealSubstrateTrainer requires a gate-issued Stage1BExecutionContext")
        self._ctx = context
        self._backend = backend                               # None → real_eegnet_trainer's default (lazy-torch) backend

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        from acar.v5.substrate import real_eegnet_trainer as RET
        if not hasattr(dataset_view, "read_label"):
            raise RealTrainerError("training requires the FIT view (with read_label); an embedding view has no labels")
        # signal + labels are FIT-only, via the view — no raw roots, no filesystem scan, no direct reader calls
        train = [(k, dataset_view.read_windows(k), dataset_view.read_label(k)) for k in train_subject_keys]
        val = [(k, dataset_view.read_windows(k), dataset_view.read_label(k)) for k in val_subject_keys]
        ref = f"{disease}/fold{fold}/seed{seed}"
        output_dir = LO.ref_output_dir(self._ctx.output_root, self._ctx.run_id, ref)   # per-ref dir via the layout helper
        backend = self._backend if self._backend is not None else RET.TorchEegnetBackend()
        return RET.train_encoder_and_source_state(disease, fold, seed, train, val, output_dir=output_dir, backend=backend)


def make_real_trainer(context):
    """Factory — construct AFTER the full-build gate, bound to the run's execution context (uses context.output_root/run_id)."""
    return RealSubstrateTrainer(context)
