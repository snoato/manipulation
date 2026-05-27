#!/bin/bash
#SBATCH --job-name=mlb-full-regen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/mlb_full_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/mlb_full_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

# Full regen of the multilevel_blocks dataset from scratch with the
# expanded primitive-exposure templates (1ca5acc), biased L5 compounds
# (cc0c5fc), and fast-mode L4 (post-Phase-3.8 — L4 no longer needs the
# full executor since put_upright IK now works reliably under fast).
#
# Produces:
#   data/multilevel_blocks/
#     train/                      600 problems (train_600 curriculum, L0-L5)
#     train_200/, train_400/      subsamples
#     val/L{0..5}/                20 each = 120 problems
#     test/L{0..5}/               20 each = 120 problems
#     train_l0_l4/                630 problems (L0-L4 only, no L5)
#
# DESTRUCTIVE: generate_mlb_curriculum.sh starts with ``rm -rf DATA_DIR``.

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DATA_DIR="${PROJECT_ROOT}/data/multilevel_blocks"
WORKERS=64

set -e
echo "========================================================================"
echo "SLURM Job: full multilevel_blocks regen (curriculum + train_l0_l4)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Started:       $(date)"
echo "Workers:       ${WORKERS}"
echo "Output dir:    ${DATA_DIR}  (will be rm -rf'd at start of step 1)"
echo "========================================================================"

PYTHON_BIN="/work/rleap1/daniel.swoboda/.conda/envs/rgnet_fresh/bin/python"
export PATH="$(dirname ${PYTHON_BIN}):${PATH}"
export MUJOCO_GL=egl

cd "${PROJECT_ROOT}"

echo
echo "--- Step 1: full curriculum (train_600 + val + test + subsamples) ---"
WORKERS=${WORKERS} bash examples/generate_mlb_curriculum.sh "${DATA_DIR}"

echo
echo "--- Step 2: train_l0_l4 (additional 630-problem OOD-train split) ---"
WORKERS=${WORKERS} bash examples/generate_mlb_train_l0_l4.sh "${DATA_DIR}"

EXIT_CODE=$?

echo "========================================================================"
echo "Finished:      $(date)"
echo "Exit code:     ${EXIT_CODE}"
echo "--- final layout ---"
for s in train train_200 train_400 train_l0_l4; do
    n=$(find "${DATA_DIR}/${s}" -name 'config_*.pddl' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "    ${s}: ${n} problems"
done
for L in 0 1 2 3 4 5; do
    for s in val test; do
        n=$(find "${DATA_DIR}/${s}/L${L}" -name 'config_*.pddl' 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && echo "    ${s}/L${L}: ${n} problems"
    done
done
echo "========================================================================"

exit ${EXIT_CODE}
