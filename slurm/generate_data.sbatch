#!/bin/bash
#SBATCH --job-name=manipulation-datagen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/datagen_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/datagen_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.swoboda@ml.rwth-aachen.de

# =============================================================================
# CONFIGURATION SECTION - Modify these variables as needed
# =============================================================================

# Project paths
PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DOMAIN_SRC="${PROJECT_ROOT}/manipulation/symbolic/domains/tabletop/pddl/domain.pddl"

# Output configuration
OUTPUT_DIR="/work/rleap1/daniel.swoboda/projects/manipulation/datasets/tabletop"
NUM_TRAIN=200
NUM_TEST=20
NUM_VAL=20

# Object configuration
MIN_OBJECTS=3
MAX_OBJECTS=7

# Grid configuration (calibrated so all cells are IK-reachable)
GRID_WIDTH=0.4
GRID_HEIGHT=0.3
CELL_SIZE=0.04
GRID_OFFSET_X=0.05
GRID_OFFSET_Y=0.25
PLACEMENT_MARGIN=1

# Motion planning / IK (benchmarked: fastest zero-false-negative)
RRT_ITERS=1000
IK_ITERS=100
IK_POS_THRESH=0.005

# Random seed (set to a specific value for reproducibility, or leave empty for random)
SEED=""

# Parallelization (0 = use all available CPUs)
NUM_WORKERS=0

# Wandb configuration
ENABLE_WANDB=true
WANDB_PROJECT="manipulation-data-generation"
WANDB_RUN_NAME="run_${SLURM_JOB_ID}"

# Visualization (set to false for faster generation on headless cluster)
GENERATE_VIZ=true

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

echo "========================================================================"
echo "SLURM Job: Manipulation Data Generation"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "========================================================================"

mkdir -p "${SLURM_SUBMIT_DIR}/logs"

# Load necessary modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate Python virtual environment
if [ -d "${PYTHON_ENV}" ]; then
    conda deactivate
    conda activate rgnet
fi

# Set environment variables for headless rendering (MuJoCo)
#export MUJOCO_GL=egl
#export DISPLAY=""

cd "${PROJECT_ROOT}" || exit 1

# Build command
CMD="python -m manipulation.symbolic.domains.tabletop.generate_data"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --num-train ${NUM_TRAIN}"
CMD="${CMD} --num-test ${NUM_TEST}"
CMD="${CMD} --num-val ${NUM_VAL}"
CMD="${CMD} --min-objects ${MIN_OBJECTS}"
CMD="${CMD} --max-objects ${MAX_OBJECTS}"
CMD="${CMD} --grid-width ${GRID_WIDTH}"
CMD="${CMD} --grid-height ${GRID_HEIGHT}"
CMD="${CMD} --cell-size ${CELL_SIZE}"
CMD="${CMD} --grid-offset-x ${GRID_OFFSET_X}"
CMD="${CMD} --grid-offset-y ${GRID_OFFSET_Y}"
CMD="${CMD} --placement-margin ${PLACEMENT_MARGIN}"
CMD="${CMD} --rrt-iters ${RRT_ITERS}"
CMD="${CMD} --ik-iters ${IK_ITERS}"
CMD="${CMD} --ik-pos-thresh ${IK_POS_THRESH}"
CMD="${CMD} --domain-src ${DOMAIN_SRC}"

if [ -n "${SEED}" ]; then
    CMD="${CMD} --seed ${SEED}"
fi

CMD="${CMD} --num-workers ${NUM_WORKERS}"

if [ "${ENABLE_WANDB}" = true ]; then
    CMD="${CMD} --wandb"
    CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        CMD="${CMD} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
fi

if [ "${GENERATE_VIZ}" = false ]; then
    CMD="${CMD} --no-viz"
fi

echo "========================================================================"
echo "Command to execute:"
echo "${CMD}"
echo "========================================================================"

eval ${CMD}

EXIT_CODE=$?

echo "========================================================================"
echo "Job finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================================================"

exit ${EXIT_CODE}
