#!/bin/bash
#SBATCH --job-name=mlb-l0_l4-datagen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/mlb_l0_l4_datagen_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/mlb_l0_l4_datagen_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

# =============================================================================
# multilevel_blocks — generate the train_l0_l4 split for OOD experiments.
# Wraps examples/generate_mlb_train_l0_l4.sh.  Non-destructive: leaves the
# existing train/, val/, test/, train_200/, train_400/ untouched and only
# adds <data_dir>/train_l0_l4/ alongside.
#
# With the LUT-seeded executor, 510 problems on 32 workers ~10-15 min.
# Budget 2h SLURM time for safety margin.
# =============================================================================

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DATA_DIR="${PROJECT_ROOT}/data/multilevel_blocks"
WORKERS=32

set -e

echo "========================================================================"
echo "SLURM Job: multilevel_blocks train_l0_l4 generation (OOD train)"
echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "Started:       $(date)"
echo "Project root:  ${PROJECT_ROOT}"
echo "Output dir:    ${DATA_DIR}/train_l0_l4"
echo "Workers/proc:  ${WORKERS}"
echo "========================================================================"

# Compute nodes don't have miniconda at the workstation's path — use the
# env's python directly via PATH prepend.  See [[reference-workstation-slurm]].
PYTHON_BIN="/work/rleap1/daniel.swoboda/.conda/envs/rgnet_fresh/bin/python"
export PATH="$(dirname ${PYTHON_BIN}):${PATH}"
export MUJOCO_GL=egl

cd "${PROJECT_ROOT}"

WORKERS=${WORKERS} bash examples/generate_mlb_train_l0_l4.sh "${DATA_DIR}"

EXIT_CODE=$?

echo "========================================================================"
echo "Finished:      $(date)"
echo "Exit code:     ${EXIT_CODE}"
echo "train_l0_l4 problems: $(find ${DATA_DIR}/train_l0_l4 -name 'config_*.pddl' 2>/dev/null | wc -l)"
echo "----- existing splits (should be unchanged) -----"
for split in train val test train_200 train_400; do
    n=$(find "${DATA_DIR}/${split}" -name 'config_*.pddl' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "    ${split}: ${n} problems"
done
echo "========================================================================"

exit ${EXIT_CODE}
