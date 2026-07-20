"""C86H gated entrypoint. Protocol/implementation preparation only.

``preflight`` is outcome-free: it verifies the §12 content-addressed bindings and the
label-blind split rule, touching no real EEG/label. ``execute`` refuses without the
direct ``授权 C86H`` token AND refuses again until the untouched confirmation field has
been generated under a separately authorized step — real Brandl2020 / ds007221 EEG,
training, prediction, label access, and active acquisition are NOT built or run here.
"""
from __future__ import annotations

import os

from oaci.theory.c86_active_program import canonical_trial_split
from . import contract as K

AUTHORIZATION_TOKEN = "授权 C86H"
# The untouched confirmation field does not exist yet; generating it is a separately
# authorized step (real EEG/label access). Its absence is the correct default state.
FIELD_ROOT = ("/projects/EEG-foundation-model/yinghao/"
              "oaci-c86h-untouched-confirmation-field-v1")


def split_target(dataset: str, subject: str, trial_ids):
    """Locked label-blind half split (§12.3); fail-closed via canonical_trial_split."""
    return canonical_trial_split(dataset, subject, trial_ids, salt=K.SPLIT_SALT)


def class_support_ok(view_labels) -> bool:
    """Post-access support gate: >= MIN_CLASS_SUPPORT labels per class per view."""
    counts = {}
    for y in view_labels:
        counts[y] = counts.get(y, 0) + 1
    return len(counts) >= 2 and all(v >= K.MIN_CLASS_SUPPORT for v in counts.values())


def preflight() -> dict:
    """Outcome-free readiness check. Opens no EEG, no labels, no target predictions."""
    bindings = K.verify_bindings()
    return {
        "stage": "C86H_PREFLIGHT",
        "authorization_present": False,
        "bindings": bindings,
        # a real field is present only if its content-addressed manifest exists (an empty stray
        # directory is not a field), so an accidental mkdir does not read as a generated field
        "field_present": os.path.isfile(os.path.join(FIELD_ROOT, "C86H_REAL_FIELD_MANIFEST.json")),
        "method_registry": list(K.METHOD_REGISTRY),
        "active_maxt_family_size": len(K.ACTIVE_METHODS) * len(K.FINITE_BUDGETS),
        "active_chains": K.ACTIVE_CHAINS,
        "materiality_margin": K.MATERIALITY_MARGIN,
        "n_targets": K.N_TARGETS,
        "ready_for_review": bindings["ok"],
        "note": ("protocol/implementation preparation only; real execution requires a "
                 "separate 授权 C86H and a separately authorized field generation"),
    }


def execute(authorization: str, output_root: str | None = None):
    """Gated confirmation trigger — delegates to the integrated runner, which refuses
    without 授权 C86H and again until the untouched field exists (separately authorized)."""
    from . import runner
    return runner.execute(authorization, output_root)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="C86H gated entrypoint (prep only)")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--authorization", default="")
    args = ap.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2, ensure_ascii=False))
    else:
        execute(args.authorization)
