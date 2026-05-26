#!/bin/bash
#SBATCH --job-name=mlb-l5-regen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/mlb_l5_regen_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/mlb_l5_regen_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

# Regenerate ONLY val/L5 and test/L5 with the L4-subsuming templates.
# Wraps examples/regenerate_eval_l5.sh.

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DATA_DIR="${PROJECT_ROOT}/data/multilevel_blocks"
WORKERS=32

set -e

echo "========================================================================"
echo "SLURM Job: regenerate L5 eval with upright-enabled templates"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "Started:       $(date)"
echo "Output dir:    ${DATA_DIR}/{val,test}/L5"
echo "========================================================================"

PYTHON_BIN="/work/rleap1/daniel.swoboda/.conda/envs/rgnet_fresh/bin/python"
export PATH="$(dirname ${PYTHON_BIN}):${PATH}"
export MUJOCO_GL=egl

cd "${PROJECT_ROOT}"

WORKERS=${WORKERS} bash examples/regenerate_eval_l5.sh "${DATA_DIR}"

EXIT_CODE=$?

echo "========================================================================"
echo "Finished:      $(date)"
echo "Exit code:     ${EXIT_CODE}"
echo "Final L5 counts:"
echo "    val/L5:  $(find ${DATA_DIR}/val/L5 -name 'config_*.pddl' 2>/dev/null | wc -l) problems"
echo "    test/L5: $(find ${DATA_DIR}/test/L5 -name 'config_*.pddl' 2>/dev/null | wc -l) problems"
echo "----- other splits (should be unchanged from prior runs) -----"
for split in train train_l0_l4 train_200 train_400; do
    n=$(find "${DATA_DIR}/${split}" -name 'config_*.pddl' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "    ${split}: ${n} problems"
done
for L in 0 1 2 3 4; do
    for split in val test; do
        n=$(find "${DATA_DIR}/${split}/L${L}" -name 'config_*.pddl' 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && echo "    ${split}/L${L}: ${n} problems"
    done
done
echo "========================================================================"

exit ${EXIT_CODE}
