#!/usr/bin/env bash
# Artifact-level gate: anonymity + legacy-framing + completeness + import/compile smoke.
# Run over the assembled reviewer artifact tree.
#   bash scripts/check_artifact.sh <artifact_root>
set -uo pipefail
ROOT="${1:?usage: check_artifact.sh <artifact_root>}"
cd "$ROOT"
fail=0
EXC=(--exclude=check_artifact.sh --exclude=check_claims.sh --exclude-dir=.git)

# identity / site strings (references.bib legitimately cites AAAI venues -> AAAI handled separately)
LEAK='W-Yinghao|Yinghao|yinwang|/home/|/projects/EEG-foundation|\binfres\b'
h=$(grep -rInE "$LEAK" . "${EXC[@]}" 2>/dev/null || true)
[ -n "$h" ] && { echo "FAIL [identity/site leak]:"; echo "$h" | head -30 | sed 's/^/    /'; fail=1; } || echo "ok   [no identity/site leaks]"

# legacy project framing
LEG='Tri-CMI|LINE CLOSED|TUAB|lockbox|CLOSEOUT'
h=$(grep -rInE "$LEG" . "${EXC[@]}" 2>/dev/null || true)
[ -n "$h" ] && { echo "FAIL [legacy framing]:"; echo "$h" | head -20 | sed 's/^/    /'; fail=1; } || echo "ok   [no legacy framing]"

# project-AAAI references (cited AAAI venues in references.bib are allowed)
h=$(grep -rInE "AAAI" . "${EXC[@]}" --exclude=references.bib --exclude=*.bbl 2>/dev/null || true)
[ -n "$h" ] && { echo "FAIL [project-AAAI reference]:"; echo "$h" | head -20 | sed 's/^/    /'; fail=1; } || echo "ok   [no project-AAAI references]"

# no submodule pointer (we vendor)
[ -f .gitmodules ] && { echo "FAIL [.gitmodules present -- vendor instead]"; fail=1; } || echo "ok   [no .gitmodules]"

# required files
for f in README.md CMI_SUBSET.md repos/TSMNet/VENDORED.md repos/TSMNet/LICENSE requirements.txt \
         tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/erasure_target_deploy_summary.json \
         tos_cmi/results/tos_cmi_eeg_frozen/factorial/factorial_multiseed.json; do
  [ -e "$f" ] && echo "ok   [present: $f]" || { echo "FAIL [missing: $f]"; fail=1; }
done
ls tos_cmi/paper/*.pdf >/dev/null 2>&1 && echo "ok   [PDF present]" || { echo "FAIL [no built PDF]"; fail=1; }

# import smoke (slim cmi subset must satisfy tos_cmi imports) + byte-compile
PY="${TOS_PY:-python}"
PYTHONPATH="$ROOT" "$PY" - <<'PYEOF' && echo "ok   [import smoke]" || { echo "FAIL [import smoke]"; fail=1; }
import importlib
for m in ["tos_cmi.eeg.erasure_target_deploy","tos_cmi.eeg.factorial_multiseed_analysis",
          "tos_cmi.eeg.erasure_baselines","tos_cmi.score_fisher",
          "cmi.paths","cmi.data.moabb_data","cmi.models.backbones","cmi.train.trainer","cmi.methods"]:
    importlib.import_module(m)
print("IMPORT_SMOKE_PASS")
PYEOF
PYTHONPATH="$ROOT" "$PY" -m compileall -q tos_cmi cmi >/dev/null 2>&1 && echo "ok   [compileall]" || { echo "FAIL [compileall]"; fail=1; }

[ "$fail" = 0 ] && echo "CHECK_ARTIFACT_PASS" || { echo "CHECK_ARTIFACT_FAIL"; exit 1; }
