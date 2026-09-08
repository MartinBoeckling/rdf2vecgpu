#!/bin/bash
#SBATCH --job-name=token_counts
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.log
# Precompute $CORPUS/token_counts.npy (unigram negative-sampling weights) ONCE
# on a CPU node, so the GPU training job loads a 3 GB file instead of having
# every torchrun rank stream the 311 GB walks corpus before step 0.
#   sbatch --export=ALL,CORPUS=$WS/corpus/wikidata_v100 performance/helix/token_counts.sbatch
set -euo pipefail

module load devel/python 2>/dev/null || true
VENV="${VENV:-$HOME/venvs/rdf2vecgpu}"
source "$VENV/bin/activate"
REPO="/gpfs/bwfor/work/ws/ma_mboeckli-rdf2vec_resource/rdf2vecgpu"
cd "$REPO"

WS="${WS:-$(ws_find rdf2vec_resource 2>/dev/null || echo "$HOME")}"
CORPUS=${CORPUS:-$WS/corpus/wikidata_v100}
echo "[token_counts] host=$(hostname -s) job=${SLURM_JOB_ID:-NA} corpus=$CORPUS cpus=${SLURM_CPUS_PER_TASK:-?}"

python performance/helix/make_token_counts.py "$CORPUS" --workers "${SLURM_CPUS_PER_TASK:-8}"
