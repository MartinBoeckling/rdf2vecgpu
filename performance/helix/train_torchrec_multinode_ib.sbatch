#!/bin/bash
#SBATCH --job-name=torchrec_mn_ib
#SBATCH --partition=gpu-multi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1            # ONE torchrun launcher per node
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.log
# Multi-node torchrec on Helix gpu4-A100 nodes over InfiniBand (2x HDR200) —
# the scaling experiment the dws script couldn't do honestly (dws inter-node is
# 10GbE TCP). gpu-multi is exclusive, gpu4 nodes only (4x A100/40GB), max 8
# nodes / 48 h => up to 32 GPUs = 1280 GB VRAM.
# Differences from performance/slurm/train_torchrec_multinode.sbatch (dws):
#   * NCCL_IB_DISABLE=0 — use the IB fabric (dws forces pure TCP).
#   * no NCCL_P2P_DISABLE (dws-A6000 PCIe workaround only).
#   * no 10.5.x NIC-subnet pinning; bootstrap iface defaults to ib0
#     (override: NCCL_SOCKET_IFNAME).
set -euo pipefail

module load devel/python 2>/dev/null || true   # venv needs libpython3.13.so.1.0
module load devel/cuda 2>/dev/null || true
VENV="${VENV:-$HOME/venvs/rdf2vecgpu}"
source "$VENV/bin/activate"
REPO="${REPO:-$(ws_find rdf2vec_resource 2>/dev/null || echo "$HOME")/rdf2vecgpu}"
cd "$REPO"

# ── rendezvous ──
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29517}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# ── NCCL: IB on ──
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_BLOCKING_WAIT=1
export RDF2VEC_TORCHREC_PROGRESS=1

# ── workload ──
WS="${WS:-$(ws_find rdf2vec_resource 2>/dev/null || echo "$HOME")}"
export CORPUS=${CORPUS:-$WS/corpus/dbpedia2016}
export OUT=${OUT:-$WS/results/mn_ib}
export EPOCHS=${EPOCHS:-1}
export PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-131072}
export LR=${LR:-0.10}
mkdir -p "$OUT"

echo "[mn-ib] nnodes=$SLURM_NNODES master=$MASTER_ADDR:$MASTER_PORT gpus/node=$GPUS_PER_NODE world=$((SLURM_NNODES*GPUS_PER_NODE))"
echo "[mn-ib] nodelist=$SLURM_JOB_NODELIST corpus=$CORPUS epochs=$EPOCHS batch=$PAIR_BATCH_SIZE"

# Pre-flight: fail fast if step launch to the second node is broken, and show
# each node's GPU/NIC topology in the log.
srun --ntasks-per-node=1 bash -c 'echo "[mn-ib] $(hostname -s) topo:"; nvidia-smi topo -m | sed -n "1,6p"'

srun --ntasks-per-node=1 bash -c '
  exec torchrun \
    --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    performance/bench_backends.py --backend torchrec \
      --corpus "$CORPUS" --out "$OUT/run.jsonl" --tag helix_mn_ib \
      --walk-depth 4 --walk-number 10 --vector-size "${VECTOR_SIZE:-200}" --window-size 5 \
      --epochs "$EPOCHS" --negative-samples 5 --min-count 1 \
      --learning-rate "$LR" --optimizer rowwiseadagrad --pair-batch-size "$PAIR_BATCH_SIZE"'
echo "[mn-ib] srun returned $?"
