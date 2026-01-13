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

# Optional: Email notifications
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=daniel.swoboda@ml.rwth-aachen.de

# =============================================================================
# CONFIGURATION SECTION - Modify these variables as needed
# =============================================================================

# Project paths
PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"  # Update this to your project path

# Output configuration
OUTPUT_DIR="/work/rleap1/daniel.swoboda/projects/rgnet/data/manipulation_dataset"  # Directory to save generated data
NUM_TRAIN=1000
NUM_TEST=100
NUM_VAL=100

# Object configuration
MIN_OBJECTS=3
MAX_OBJECTS=7

# Grid configuration
GRID_WIDTH=0.4
GRID_HEIGHT=0.3
CELL_SIZE=0.04
GRID_OFFSET_Y=0.02

# Random seed (set to a specific value for reproducibility, or leave empty for random)
SEED=""

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

# Create logs directory if it doesn't exist
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

# Load necessary modules (adjust based on your cluster)
# Examples:
# module load cuda/11.8
# module load python/3.10
# module load mujoco/2.3.0

# Activate Python virtual environment
if [ -d "${PYTHON_ENV}" ]; then
    conda deactivate
    conda activate rgnet
fi

# Set environment variables for headless rendering (MuJoCo)
#export MUJOCO_GL=egl  # Use EGL for GPU rendering without display
#export DISPLAY=""     # Disable X11 display

# Optionally set MuJoCo license path if needed
# export MJKEY_PATH="/path/to/mjkey.txt"

# Navigate to project root
cd "${PROJECT_ROOT}" || exit 1

# Build command with arguments
CMD="python examples/generate_data.py"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --num-train ${NUM_TRAIN}"
CMD="${CMD} --num-test ${NUM_TEST}"
CMD="${CMD} --num-val ${NUM_VAL}"
CMD="${CMD} --min-objects ${MIN_OBJECTS}"
CMD="${CMD} --max-objects ${MAX_OBJECTS}"
CMD="${CMD} --grid-width ${GRID_WIDTH}"
CMD="${CMD} --grid-height ${GRID_HEIGHT}"
CMD="${CMD} --cell-size ${CELL_SIZE}"
CMD="${CMD} --grid-offset-y ${GRID_OFFSET_Y}"

# Add seed if specified
if [ -n "${SEED}" ]; then
    CMD="${CMD} --seed ${SEED}"
fi

# Add wandb flags if enabled
if [ "${ENABLE_WANDB}" = true ]; then
    CMD="${CMD} --wandb"
    CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
    
    if [ -n "${WANDB_RUN_NAME}" ]; then
        CMD="${CMD} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
    
fi

# Disable visualization if requested
if [ "${GENERATE_VIZ}" = false ]; then
    CMD="${CMD} --no-viz"
fi

echo "========================================================================"
echo "Command to execute:"
echo "${CMD}"
echo "========================================================================"

# Execute the command
eval ${CMD}

# Capture exit code
EXIT_CODE=$?

echo "========================================================================"
echo "Job finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================================================"

# Exit with the same code as the Python script
exit ${EXIT_CODE}
