#!/bin/bash
# Regenerate val/L{0,1,2} + test/L{0,1,2} after the primitive-exposure
# template additions (1ca5acc).  Leaves val/L3, val/L4, val/L5, test/L3..5,
# train, train_l0_l4 untouched.

set -e

DATA_DIR="${1:-data/multilevel_blocks}"
WORKERS="${WORKERS:-32}"

gen_level() {
    local split="$1"     # val | test
    local level="$2"     # 0 | 1 | 2
    local spec="${split}_per_level"
    echo
    echo "=== $spec  split=$split  level=L$level ==="
    rm -rf "$DATA_DIR/$split/L$level"
    python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
        --output-dir "$DATA_DIR" \
        --curriculum-spec "$spec" \
        --curriculum-split "$split" \
        --level "$level" \
        --config-offset 0 \
        --num-workers "$WORKERS"
}

for level in 0 1 2; do
    gen_level val  "$level"
    gen_level test "$level"
done

echo
echo "=== DONE ==="
for split in val test; do
    for level in 0 1 2; do
        n=$(find "$DATA_DIR/$split/L$level" -name 'config_*.pddl' 2>/dev/null | wc -l)
        echo "  $split/L$level: $n problems"
    done
done
