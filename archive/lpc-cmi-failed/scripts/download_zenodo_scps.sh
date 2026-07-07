#!/bin/bash
# Download SCPS Zenodo cohorts (records with data >0.1GB). No GPU. Updates MANIFEST status.
set -u
DEST=/projects/EEG-foundation-model/datalake/raw/scps
LOG=$DEST/download_zenodo.log
# id:cond  (skipping 0-byte/PDF-only: 3898010,3898024 ALS, 7152316 ADHD)
LIST=(4316608:AD 7428378:AD 6225534:AD 10683141:ASD 7149276:ASD 5002434:ASD 4501988:ALS 4002038:ALS 29601:SCZ)
echo "$(date) START Zenodo download (${#LIST[@]} records, ~46GB)" | tee -a "$LOG"
for e in "${LIST[@]}"; do
  id=${e%%:*}; cond=${e##*:}; out=$DEST/$cond/zenodo_$id; mkdir -p "$out"
  echo "$(date) >>> zenodo $id ($cond) -> $out" | tee -a "$LOG"
  # get (url \t filename) from API
  curl -s --max-time 30 "https://zenodo.org/api/records/$id" | \
    /home/infres/yinwang/anaconda3/bin/python3 -c "import sys,json
d=json.load(sys.stdin)
for f in d.get('files',[]):
    u=f.get('links',{}).get('self',''); k=f.get('key','')
    if u and k: print(u+'\t'+k)" > "$out/.urls" 2>/dev/null
  while IFS=$'\t' read -r url fn; do
    [ -z "$url" ] && continue
    [ -f "$out/$fn" ] && { echo "  have $fn"; continue; }
    echo "$(date)   wget $fn" | tee -a "$LOG"
    wget -q --tries=3 --timeout=60 -O "$out/$fn" "$url" 2>>"$LOG" || echo "$(date)   FAIL $fn" | tee -a "$LOG"
  done < "$out/.urls"
  sz=$(du -sh "$out" 2>/dev/null | awk '{print $1}'); echo "$(date) DONE zenodo $id ($sz)" | tee -a "$LOG"
done
echo "$(date) ALL Zenodo DONE" | tee -a "$LOG"
