#!/bin/bash
# Generate the L0-L4 train split (no L5, held out for OOD evaluation).
#
# NON-DESTRUCTIVE: this driver does NOT rm -rf the data dir.  It only
# adds <data_dir>/train_l0_l4/ alongside the existing train/, val/, test/.
# If <data_dir>/train_l0_l4/ already exists it'll be overwritten by the
# per-level generate_data calls; clear it first if you need a clean slate.
#
# Usage:
#   ./examples/generate_mlb_train_l0_l4.sh [data_dir]
#
# Defaults: data_dir = data/multilevel_blocks/, WORKERS=32 (override via env).
# Outputs: <data_dir>/train_l0_l4/config_{0..509}.pddl + .plan + viz/

set -e

DATA_DIR="${1:-data/multilevel_blocks}"
WORKERS="${WORKERS:-32}"

mkdir -p "$DATA_DIR"

# Per-curriculum-spec, per-level driver — mirrors generate_mlb_curriculum.sh
# but writes only the L0-L4 train split.
gen_one() {
    local level="$1"     # 0..4
    local offset="$2"    # config_num start (for flat output)
    echo
    echo "=== train_l0_l4  split=train_l0_l4  level=L$level  offset=$offset ==="
    python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
        --output-dir "$DATA_DIR" \
        --curriculum-spec train_l0_l4 \
        --curriculum-split train_l0_l4 \
        --level "$level" \
        --config-offset "$offset" \
        --num-workers "$WORKERS"
}

# train_l0_l4 (flat): 60 + 120 + 120 + 120 + 90 = 510 problems
gen_one 0 0
gen_one 1 60
gen_one 2 180
gen_one 3 300
gen_one 4 420

echo
echo "=== DONE ==="
echo "Output: $DATA_DIR/train_l0_l4"
n=$(find "$DATA_DIR/train_l0_l4" -name 'config_*.pddl' 2>/dev/null | wc -l)
echo "Problems: $n"
