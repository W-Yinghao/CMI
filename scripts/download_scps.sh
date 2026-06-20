#!/bin/bash
# Download SCPS cross-dataset OpenNeuro cohorts (EEG only) into the datalake raw store.
# No GPU. Skips MEG/fMRI-bloat datasets. Updates MANIFEST.tsv status. Guards disk space.
set -u
DEST=/projects/EEG-foundation-model/datalake/raw/scps
MAN=$DEST/MANIFEST.tsv
LOG=$DEST/download.log
S3="aws s3 sync --no-sign-request --only-show-errors"
COMMON_EXCL=(--exclude "derivatives/*" --exclude ".datalad/*" --exclude ".git/*" --exclude "sourcedata/*")

mark(){ # mark dsid status in manifest
  python3 - "$MAN" "$1" "$2" <<'PY'
import sys
man,key,st=sys.argv[1:4]
L=open(man).read().splitlines()
out=[]
for ln in L:
    if ("\t"+key+"\t") in ("\t"+ln+"\t") and key in ln and ln.endswith("PENDING"):
        ln="\t".join(ln.split("\t")[:-1])+"\t"+st
    out.append(ln)
open(man,"w").write("\n".join(out)+"\n")
PY
}
guard(){ # abort if < 150 GB free
  free=$(df -BG --output=avail "$DEST" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt 150 ]; then echo "$(date) ABORT: only ${free}GB free" | tee -a "$LOG"; exit 1; fi
}

# (dsid cond mode) — priority order: PD flagship first, then AD/DEP/SCZ/ASD-eeg/ALS/ADHD
LIST=(
  "ds002778 PD full" "ds004584 PD full" "ds003490 PD full" "ds004152 PD full"
  "ds001787 PD full" "ds004315 PD full" "ds004574 PD full" "ds004580 PD full"
  "ds003506 PD full" "ds003509 PD full"
  "ds004504 AD full" "ds003800 AD full" "ds004796 AD eegonly"
  "ds003478 DEP full" "ds003474 DEP full"
  "ds003944 SCZ full" "ds003947 SCZ full" "ds004000 SCZ full" "ds004367 SCZ full"
  "ds005406 ASD full"
)
echo "$(date) START SCPS OpenNeuro download (20 datasets, ~295GB)" | tee -a "$LOG"
for entry in "${LIST[@]}"; do
  set -- $entry; dsid=$1; cond=$2; mode=$3
  out=$DEST/$cond/$dsid
  if [ -d "$out" ] && [ -f "$out/dataset_description.json" ]; then
    echo "$(date) SKIP $dsid (already present)" | tee -a "$LOG"; mark "$dsid" "DONE(prior)"; continue
  fi
  guard
  mkdir -p "$out"
  echo "$(date) >>> $dsid ($cond, $mode) -> $out" | tee -a "$LOG"
  if [ "$mode" = "eegonly" ]; then
    $S3 "s3://openneuro.org/$dsid/" "$out/" --exclude "*" \
        --include "*/eeg/*" --include "*.json" --include "*.tsv" --include "*.bval" 2>>"$LOG"
  else
    $S3 "s3://openneuro.org/$dsid/" "$out/" "${COMMON_EXCL[@]}" 2>>"$LOG"
  fi
  rc=$?
  sz=$(du -sh "$out" 2>/dev/null | awk '{print $1}')
  if [ $rc -eq 0 ]; then echo "$(date) DONE $dsid ($sz)" | tee -a "$LOG"; mark "$dsid" "DONE($sz)";
  else echo "$(date) FAIL $dsid rc=$rc" | tee -a "$LOG"; mark "$dsid" "FAIL"; fi
done
echo "$(date) ALL OpenNeuro DONE. free=$(df -BG --output=avail "$DEST"|tail -1)" | tee -a "$LOG"
# mark the two intentional skips
mark "ds005234" "SKIP(MEG)"; mark "ds003823" "SKIP(fMRI-biofeedback)"
