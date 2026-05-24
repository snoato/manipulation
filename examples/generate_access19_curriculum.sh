#!/usr/bin/env bash
# Shell driver — one Python process per (split, level) tuple.
#
# Why: macOS spawn-pool reuse hangs after the first pool.close(); using
# a fresh Python process per level avoids that and also gives us per-
# level resource caps.  Mirrors examples/generate_mlb_curriculum.sh.
set -euo pipefail

OUT_DIR="${OUT_DIR:-data/access19}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
CURRICULUM="${CURRICULUM:-train_120}"
TIME_BUDGET="${TIME_BUDGET:-180}"

PY="${PY:-python}"
GEN="${PY} -m tampanda.symbolic.domains.access19.generate_data"

# train levels — counts per curriculum.
case "$CURRICULUM" in
  train_120) COUNTS=(16 24 32 32 16) ;;
  *) echo "unknown curriculum: $CURRICULUM" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"

offset=0
for level in 0 1 2 3 4; do
  count=${COUNTS[$level]}
  [ "$count" -eq 0 ] && continue
  echo "=== train L${level}  ${count} instances ==="
  $GEN --output-dir "$OUT_DIR" \
       --num-workers "$NUM_WORKERS" \
       --seed "$SEED" \
       --time-budget "$TIME_BUDGET" \
       --level "$level" --num "$count" --split train \
       --config-offset "$offset" \
       --no-domain-copy
  offset=$(( offset + count ))
done

# val + test — same counts per level (6 each).
for split in val test; do
  offset=0
  for level in 0 1 2 3 4; do
    echo "=== ${split} L${level}  6 instances ==="
    $GEN --output-dir "$OUT_DIR" \
         --num-workers "$NUM_WORKERS" \
         --seed $(( SEED + 100 + (split == "test" ? 100 : 0) )) \
         --time-budget "$TIME_BUDGET" \
         --level "$level" --num 6 --split "$split" \
         --config-offset "$offset" \
         --no-domain-copy
    offset=$(( offset + 6 ))
  done
done

# Copy domain.pddl once.
$PY -c "
from pathlib import Path
import shutil
from tampanda.symbolic.domains.access19.generate_data import _DOMAIN
shutil.copy(_DOMAIN, Path('$OUT_DIR') / 'domain.pddl')
print('Copied domain.pddl to $OUT_DIR/')
"

echo "=== curriculum complete: $OUT_DIR ==="
