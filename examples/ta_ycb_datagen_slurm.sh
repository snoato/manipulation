#!/bin/bash
#SBATCH --job-name=ta-ycb-datagen
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/ta_ycb_datagen_%A_%a.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/ta_ycb_datagen_%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH --partition=rleap_gpu_24gb

# tabletop_access_ycb dataset generation, sharded across a SLURM array.
# Data-gen is CPU-only (MuJoCo physics + IK, no rendering — no GPU needed).
# Submit onto a CPU partition by overriding the directive above, e.g.:
#   sbatch --partition=<rleap_cpu_partition> examples/ta_ycb_datagen_slurm.sh
#
# Each task generates a proportional shard of every split with a distinct
# seed and a non-colliding --start-index, all written into the shared
# data/access_ycb/{train,val,eval_ood} dirs (domain.pddl is identical per
# task).  Every plan is FULL-validated (default --full-check-frac 1.0), so
# every written plan is FULL-executable.
#
# Totals across the array (N_TASKS shards): train 300 / val 50 / eval 50.

set -e

N_TASKS=10
PER_TRAIN=30
PER_VAL=5
PER_EVAL=5

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
OUT_DIR="${PROJECT_ROOT}/data/access_ycb"
PYTHON_BIN="/work/rleap1/daniel.swoboda/.conda/envs/rgnet_fresh/bin/python"

TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((1000 + TASK))

echo "========================================================================"
echo "ta_ycb datagen shard ${TASK}/${N_TASKS}  job=${SLURM_JOB_ID:-local}"
echo "  out=${OUT_DIR}  seed=${SEED}"
echo "  train=${PER_TRAIN}@$((TASK*PER_TRAIN))  val=${PER_VAL}@$((TASK*PER_VAL))  eval=${PER_EVAL}@$((TASK*PER_EVAL))"
echo "  started $(date)"
echo "========================================================================"

export PATH="$(dirname ${PYTHON_BIN}):${PATH}"
# Physics-only (no rendering) — no GL backend required.
# Pre-populated shared-FS YCB cache (compute nodes have no ~/.cache / internet);
# populate once on a workstation: TAMPANDA_ASSETS_CACHE=<path> python -c \
#   "from ...tabletop_access_ycb.setup import build_setup; build_setup(__import__('tempfile').mkdtemp())"
export TAMPANDA_ASSETS_CACHE=/work/rleap1/daniel.swoboda/tampanda_assets_cache
cd "${PROJECT_ROOT}"

# One --start-index (= TASK * PER_TRAIN) offsets every split's NNNN.  Since
# PER_TRAIN is the largest per-split count, the per-split index windows never
# overlap across shards (val/eval just get sparse, gap'd numbering — harmless;
# the loader globs *.pddl).
OFFSET=$((TASK * PER_TRAIN))

"${PYTHON_BIN}" -m tampanda.symbolic.domains.tabletop_access_ycb.generate_data \
    --output-dir "${OUT_DIR}" \
    --train ${PER_TRAIN} --val ${PER_VAL} --eval ${PER_EVAL} \
    --seed ${SEED} \
    --start-index ${OFFSET}

echo "shard ${TASK} done $(date)"
