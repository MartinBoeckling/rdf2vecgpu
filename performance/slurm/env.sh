# Shared environment for the rdf2vecgpu cross-backend benchmark jobs.
# Sourced (not executed) by every train_*.sbatch / prepare.sbatch.
#
# Expects the following to be exported by the submitting job (see submit_sweep.sh):
#   DATA CORPUS OUTDIR TAG BACKEND
#   WALK_STRATEGY WALK_DEPTH WALK_NUMBER EMB_MODEL EPOCHS VECTOR_SIZE
#   WINDOW MIN_COUNT LR NEGATIVE RANDOM_STATE CPU_COUNT

# Activate the conda env BEFORE `set -u` — conda's own scripts trip over unbound vars.
source /ceph/mboeckli/miniconda3/etc/profile.d/conda.sh
conda activate rdf2vecgpu_test
set -euo pipefail

export REPO=/ceph/mboeckli/rdf2vecgpu
export WORK=/ceph/mboeckli/rdf2vec_bench
cd "$REPO"

# Job-local scratch so cudf/parquet temp files don't fill /tmp.
export TMPDIR="$WORK/scratch/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" "$OUTDIR"

# Training hyperparameters, identical across every backend (apples-to-apples).
# A bash array so each job can splat it: python ... "${HP_ARGS[@]}"
HP_ARGS=(
  --walk-strategy   "$WALK_STRATEGY"
  --walk-depth      "$WALK_DEPTH"
  --walk-number     "$WALK_NUMBER"
  --embedding-model "$EMB_MODEL"
  --epochs          "$EPOCHS"
  --vector-size     "$VECTOR_SIZE"
  --window-size     "$WINDOW"
  --min-count       "$MIN_COUNT"
  --learning-rate   "$LR"
  --negative-samples "$NEGATIVE"
  --random-state    "$RANDOM_STATE"
  --cpu-count       "$CPU_COUNT"
  --grad-clip-value "$GRAD_CLIP"
  --optimizer       "$OPTIMIZER"
)

echo "[env] host=$(hostname) job=${SLURM_JOB_ID:-NA} backend=${BACKEND:-?} tag=${TAG:-?}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} corpus=$CORPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
