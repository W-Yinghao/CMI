#!/bin/bash
cd /home/infres/yinwang/CMI_AAAI_readout_prior
for i in $(seq 1 720); do
  n=$(ls results/precache_done/*.done 2>/dev/null | wc -l)
  [ "$n" -ge 91 ] && { echo "EPOCH_CACHE_COMPLETE 91/91"; exit 0; }
  sleep 30
done
echo "EPOCH_WAIT_TIMEOUT $(ls results/precache_done/*.done 2>/dev/null | wc -l)/91"; exit 1
