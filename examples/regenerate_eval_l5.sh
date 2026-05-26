#!/bin/bash
# Regenerate ONLY val/L5/ and test/L5/ with the L4-subsuming templates
# (compound now allows upright_bridges + tower_on_bridge substructures,
# and double_bridges is now sampled in L5).
#
# Used for the OOD compositional generalisation eval: training on
# train_l0_l4/ never sees L5 templates; the freshly-regenerated L5 val
# + test contain the FULL L5 vocabulary including the upright family.
#
# NON-DESTRUCTIVE for everything except val/L5 and test/L5.  Other
# splits (train, train_l0_l4, train_200, train_400, val/L0..L4,
# test/L0..L4) are untouched.

set -e

DATA_DIR="${1:-data/multilevel_blocks}"
WORKERS="${WORKERS:-32}"

echo "Regenerating val/L5 and test/L5 in $DATA_DIR (uprights enabled)"

# Clear ONLY the L5 eval subdirs.
rm -rf "$DATA_DIR/val/L5" "$DATA_DIR/test/L5"

# val_per_level → val/L5/
echo
echo "=== val_per_level  split=val  level=L5 ==="
python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
    --output-dir "$DATA_DIR" \
    --curriculum-spec val_per_level \
    --curriculum-split val \
    --level 5 \
    --config-offset 0 \
    --num-workers "$WORKERS"

# test_per_level → test/L5/
echo
echo "=== test_per_level  split=test  level=L5 ==="
python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
    --output-dir "$DATA_DIR" \
    --curriculum-spec test_per_level \
    --curriculum-split test \
    --level 5 \
    --config-offset 0 \
    --num-workers "$WORKERS"

echo
echo "=== DONE ==="
val_n=$(find "$DATA_DIR/val/L5" -name 'config_*.pddl' 2>/dev/null | wc -l)
test_n=$(find "$DATA_DIR/test/L5" -name 'config_*.pddl' 2>/dev/null | wc -l)
echo "  val/L5:  $val_n problems"
echo "  test/L5: $test_n problems"
echo
echo "Quick template breakdown:"
for split in val test; do
    echo "  $split/L5 templates:"
    for f in "$DATA_DIR/$split/L5"/*.plan; do
        grep -h "^; template:" "$f" 2>/dev/null
    done | sort | uniq -c | sort -rn | head
done
