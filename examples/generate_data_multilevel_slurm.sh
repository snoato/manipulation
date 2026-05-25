#!/bin/bash
#SBATCH --job-name=mlb-datagen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/mlb_datagen_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/mlb_datagen_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

# =============================================================================
# multilevel_blocks dataset generation — full curriculum (train_600 + val/test).
#
# Wraps examples/generate_mlb_curriculum.sh, which drives one Python process
# per (split, level) pair to sidestep the fork-pool reuse hang.  With the
# Phase-3.7 LUT-seeded executor, the full curriculum takes <2 h with 32
# workers per process (was many hours pre-LUT).
#
# Output: $PROJECT_ROOT/data/multilevel_blocks/
#   train/       (flat — train_600 mix across L0-L5, also train_200/train_400 subsamples)
#   val/L{0..5}/
#   test/L{0..5}/
#
# Re-run safety: examples/generate_mlb_curriculum.sh DOES rm -rf the output
# dir first.  Make sure nothing under data/multilevel_blocks/ is irreplaceable
# before launching.
# =============================================================================

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DATA_DIR="${PROJECT_ROOT}/data/multilevel_blocks"
WORKERS=32

set -e

echo "========================================================================"
echo "SLURM Job: multilevel_blocks dataset generation"
echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "Started:       $(date)"
echo "Project root:  ${PROJECT_ROOT}"
echo "Output dir:    ${DATA_DIR}"
echo "Workers/proc:  ${WORKERS}"
echo "========================================================================"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate rgnet_fresh

# Headless rendering — MuJoCo will try to open a GL context for env init even
# when --no-viz is set inside generate_data.py (it isn't, by default, so PNG
# previews are rendered too).
export MUJOCO_GL=egl

cd "${PROJECT_ROOT}"

WORKERS=${WORKERS} bash examples/generate_mlb_curriculum.sh "${DATA_DIR}"

EXIT_CODE=$?

echo "========================================================================"
echo "Finished:      $(date)"
echo "Exit code:     ${EXIT_CODE}"
echo "Final size:    $(du -sh ${DATA_DIR} 2>/dev/null | cut -f1)"
echo "Problem counts:"
for split in train val test; do
    n=$(find "${DATA_DIR}/${split}" -name "config_*.pddl" 2>/dev/null | wc -l)
    echo "    ${split}: ${n} problems"
done
echo "========================================================================"

exit ${EXIT_CODE}
