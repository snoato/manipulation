#!/bin/bash
# Drive the multilevel_blocks curriculum generation as one-Python-process-
# per-level + per-split.  This sidesteps the fork-pool reuse hang we hit
# when one process drives multiple sequential mp.Pool instances on macOS.
#
# Usage: ./examples/generate_mlb_curriculum.sh [data_dir]
# Defaults to data/multilevel_blocks/.

set -e

DATA_DIR="${1:-data/multilevel_blocks}"
WORKERS="${WORKERS:-4}"

rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# Per-curriculum-spec, per-level driver.
gen_one() {
    local spec="$1"      # e.g., train_600
    local split="$2"     # e.g., train
    local level="$3"     # 0..5
    local offset="$4"    # config_num start (for flat train)
    echo
    echo "=== $spec  split=$split  level=L$level  offset=$offset ==="
    python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
        --output-dir "$DATA_DIR" \
        --curriculum-spec "$spec" \
        --curriculum-split "$split" \
        --level "$level" \
        --config-offset "$offset" \
        --num-workers "$WORKERS"
}

# train_600 (flat): levels mixed in train/ with offset per level.
#   60 + 120 + 120 + 120 + 90 + 90 = 600
gen_one train_600 train 0 0
gen_one train_600 train 1 60
gen_one train_600 train 2 180
gen_one train_600 train 3 300
gen_one train_600 train 4 420
gen_one train_600 train 5 510

# val_per_level + test_per_level: each level into its own subdir;
# offset doesn't matter since per-level subdirs are separate.
for level in 0 1 2 3 4 5; do
    gen_one val_per_level val "$level" 0
done
for level in 0 1 2 3 4 5; do
    gen_one test_per_level test "$level" 0
done

# Subsample train to 200 and 400.
echo
echo "=== train_200 subsample ==="
python -u examples/sample_dataset_subset.py \
    --source "$DATA_DIR/train" \
    --dest "$DATA_DIR/train_200" \
    --size 200 --seed 42

echo
echo "=== train_400 subsample ==="
python -u examples/sample_dataset_subset.py \
    --source "$DATA_DIR/train" \
    --dest "$DATA_DIR/train_400" \
    --size 400 --seed 42

echo
echo "=== DONE ==="
echo "Output: $DATA_DIR"
