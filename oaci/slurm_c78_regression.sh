#!/usr/bin/env bash
#SBATCH --job-name=c78-reg
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONDONTWRITEBYTECODE=1
suite="${1:?suite required}"
case "$suite" in
  focused)
    python -m pytest -p no:cacheprovider oaci/tests/test_c78_seed3_instrumented_pilot.py -q
    ;;
  c65)
    python -m pytest -p no:cacheprovider oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-8]_*.py -q
    ;;
  c23)
    python -m pytest -p no:cacheprovider oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6][0-9]_*.py oaci/tests/test_c7[0-8]_*.py -q
    ;;
  full)
    python -m pytest -p no:cacheprovider oaci/tests -q
    ;;
  *)
    echo "unknown suite: $suite" >&2
    exit 2
    ;;
esac
