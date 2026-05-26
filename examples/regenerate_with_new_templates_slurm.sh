#!/bin/bash
#SBATCH --job-name=mlb-newprim-regen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/mlb_newprim_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/mlb_newprim_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

# Regenerate after the primitive-exposure template additions (commit 1ca5acc):
#   * train_l0_l4 (630 problems, bumped L0/L1 quotas)
#   * val/L{0,1,2} + test/L{0,1,2} (new templates only land in these levels)
# Leaves val/L{3,4,5}, test/L{3,4,5}, train/, train_200/, train_400/ alone.

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DATA_DIR="${PROJECT_ROOT}/data/multilevel_blocks"
WORKERS=32

set -e
echo "========================================================================"
echo "SLURM Job: regenerate train_l0_l4 + val/test L0-L2 (new primitive templates)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Started:       $(date)"
echo "Output dir:    ${DATA_DIR}"
echo "Workers:       ${WORKERS}"
echo "========================================================================"

PYTHON_BIN="/work/rleap1/daniel.swoboda/.conda/envs/rgnet_fresh/bin/python"
export PATH="$(dirname ${PYTHON_BIN}):${PATH}"
export MUJOCO_GL=egl

cd "${PROJECT_ROOT}"

echo
echo "--- Step 1: train_l0_l4 (630 problems with bumped L0/L1 quotas) ---"
WORKERS=${WORKERS} bash examples/generate_mlb_train_l0_l4.sh "${DATA_DIR}"

echo
echo "--- Step 2: val + test for L0/L1/L2 (where new templates appear) ---"
WORKERS=${WORKERS} bash examples/regenerate_eval_l0_l1_l2.sh "${DATA_DIR}"

EXIT_CODE=$?

echo "========================================================================"
echo "Finished:      $(date)"
echo "Exit code:     ${EXIT_CODE}"
echo "----- final layout -----"
for split in train train_200 train_400 train_l0_l4; do
    n=$(find "${DATA_DIR}/${split}" -name 'config_*.pddl' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "    ${split}: ${n} problems"
done
for L in 0 1 2 3 4 5; do
    for split in val test; do
        n=$(find "${DATA_DIR}/${split}/L${L}" -name 'config_*.pddl' 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && echo "    ${split}/L${L}: ${n} problems"
    done
done
echo "========================================================================"

exit ${EXIT_CODE}
