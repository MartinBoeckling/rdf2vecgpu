#!/bin/bash
#SBATCH --job-name=torchrec_gpu8
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=%x_%j.log
# Single-node torchrec training on a Helix gpu8 node (8x A100/80GB = 640 GB
# VRAM, or 8x H200/141GB = 1128 GB; 2.2 TB RAM; gpu-single allows 120 h).
# Header defaults are the SMOKE shape (2x A100, dbpedia2016, 1 epoch). Scale up
# at submit time — full-Wikidata d100 fits ONE A100-gpu8 node (314 GB tables ->
# ~39 GB/GPU), d200 needs the H200 flavor (629 GB -> ~79 GB/GPU):
#   sbatch --gres=gpu:A100:8 --cpus-per-task=48 --mem=800G --time=96:00:00 \
#     --export=ALL,GPUS=8,CORPUS=$WS/corpus/wikidata_v100,EPOCHS=5,... this-file
# Single node => torchrun --standalone, NO multi-node rendezvous, and the
# all-to-all stays on NVLink/NVSwitch. Do NOT set NCCL_P2P_DISABLE here (that
# flag is the dws-A6000 PCIe workaround; on NVLink boxes it disables the fast
# path).
set -euo pipefail

module load devel/python 2>/dev/null || true   # venv needs libpython3.13.so.1.0
module load devel/cuda 2>/dev/null || true
VENV="${VENV:-$HOME/venvs/rdf2vecgpu}"
source "$VENV/bin/activate"
REPO="/gpfs/bwfor/work/ws/ma_mboeckli-rdf2vec_resource/rdf2vecgpu"
cd "$REPO"

export GPUS=${GPUS:-2}
export RDF2VEC_TORCHREC_PROGRESS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── workload ──
WS="${WS:-$(ws_find rdf2vec_resource 2>/dev/null || echo "$HOME")}"
export CORPUS=${CORPUS:-$WS/corpus/dbpedia2016}
export OUT=${OUT:-$WS/results/gpu8}
export EPOCHS=${EPOCHS:-1}
export PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-131072}
export LR=${LR:-0.10}
mkdir -p "$OUT"

echo "[gpu8] host=$(hostname -s) gpus=$GPUS corpus=$CORPUS epochs=$EPOCHS batch=$PAIR_BATCH_SIZE lr=$LR"
nvidia-smi topo -m | sed -n "1,$((GPUS+2))p"

torchrun --standalone --nproc_per_node="$GPUS" \
  performance/bench_backends.py --backend torchrec \
    --corpus "$CORPUS" --out "$OUT/run.jsonl" --tag helix_gpu8 \
    --walk-depth 4 --walk-number 10 --vector-size "${VECTOR_SIZE:-200}" --window-size 5 \
    --epochs "$EPOCHS" --negative-samples 5 --min-count 1 \
    --learning-rate "$LR" --optimizer rowwiseadagrad --pair-batch-size "$PAIR_BATCH_SIZE" \
    ${EXPORT_KGREAT:+--export-kgreat "$EXPORT_KGREAT"}
echo "[gpu8] torchrun returned $?"
