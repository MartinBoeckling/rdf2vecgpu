#!/bin/bash
#SBATCH --job-name=torchrec_mn_h100
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1            # ONE torchrun launcher per node; it spawns the GPU workers
#SBATCH --gpus-per-socket=2            # NVLinked pair per socket (NOT --gres: a generic
#SBATCH --sockets-per-node=2           # request may hand out cross-socket GPUs with no NVLink)
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# H100 variant of train_torchrec_multinode.sbatch. Differences from the A6000 one:
#   * gpu-vram-94gb (dws-12/13): each CPU socket hosts an H100 NVL pair bridged
#     with 600 GB/s NVLink (nvidia-smi topo shows NV12). SLURM only guarantees a
#     bridged pair via --gpus-per-socket=2, per admin guidance — hence the
#     per-socket request above (2 sockets x 2 GPUs = 4 GPUs/node).
#   * NO NCCL_P2P_DISABLE: that flag is the A6000/PCIe hang workaround and on
#     H100 it would switch the NVLink bridges off. NCCL detects NV12 and routes
#     intra-pair traffic over it when P2P is allowed.
# Everything else (TCP inter-node over the 10.5.x net, subnet-pinned NIC,
# workload defaults) matches the A6000 script.
#
# Override defaults via env: GPUS_PER_NODE, CORPUS, EPOCHS, PAIR_BATCH_SIZE, LR,
# EXPORT_KGREAT.
set -euo pipefail

source /ceph/mboeckli/miniconda3/etc/profile.d/conda.sh
conda activate rdf2vecgpu_test
export REPO=/ceph/mboeckli/rdf2vecgpu
cd "$REPO"

# ── rendezvous ──
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29517}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# ── NCCL: P2P stays ON (NVLink pairs — see header). Inter-node = TCP on the
#    10.5.x cluster net (no IB on dws); _mn_launch.sh pins NCCL_SOCKET_IFNAME by
#    IP subnet per node since NIC names differ across nodes.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IFACE_SUBNET=${NCCL_IFACE_SUBNET:-10.5.}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_BLOCKING_WAIT=1
export RDF2VEC_TORCHREC_PROGRESS=1

# ── workload (dbpedia2016 corpus already on disk) ──
export CORPUS=${CORPUS:-/ceph/mboeckli/rdf2vec_bench/corpus/dbpedia2016}
export OUT=${OUT:-/ceph/mboeckli/rdf2vec_bench/results/mn_test_h100}
export EPOCHS=${EPOCHS:-1}
export PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-131072}
export LR=${LR:-0.05}
mkdir -p "$OUT"

echo "[mn] nnodes=$SLURM_NNODES master=$MASTER_ADDR:$MASTER_PORT gpus/node=$GPUS_PER_NODE world=$((SLURM_NNODES*GPUS_PER_NODE))"
echo "[mn] nodelist=$SLURM_JOB_NODELIST corpus=$CORPUS epochs=$EPOCHS batch=$PAIR_BATCH_SIZE"

# Pre-flight: verify the NVLink pairing on every node (expect NV12 between the
# socket pair) and fail fast if the multi-node step launch itself is broken
# (seen on dws-15+17: "timeout waiting for task launch" before any code ran).
srun --ntasks-per-node=1 bash -c 'echo "[mn] $(hostname -s) topo:"; nvidia-smi topo -m | sed -n "1,5p"'

# One torchrun per node (helper resolves per-node SLURM_NODEID + NIC). Env is
# propagated to the srun tasks automatically.
srun --ntasks-per-node=1 bash "$REPO/performance/slurm/_mn_launch.sh"
echo "[mn] srun returned $?"
