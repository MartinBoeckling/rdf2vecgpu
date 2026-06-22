#!/bin/bash
#SBATCH --job-name=torchrec_mn
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1            # ONE torchrun launcher per node; it spawns the GPU workers
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# Multi-node torchrec end-to-end smoke test. Proves the launcher mechanics:
# cross-node rendezvous, DMP sharding across the full (2-node) process group, the
# per-step all-to-all over the inter-node link, and embedding export. dws has NO
# InfiniBand, so NCCL runs over TCP (enp194s0f0np0, the 10.5.x cluster net) —
# slow but correct; the IB-fast run is for Helix/bwUniCluster.
#
# Override defaults via env: NODES (info only — set --nodes too), GPUS_PER_NODE,
# CORPUS, EPOCHS, PAIR_BATCH_SIZE, LR, EXPORT_KGREAT.
set -euo pipefail

source /ceph/mboeckli/miniconda3/etc/profile.d/conda.sh
conda activate rdf2vecgpu_test
export REPO=/ceph/mboeckli/rdf2vecgpu
cd "$REPO"

# ── rendezvous ──
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29517}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# ── NCCL: A6000 has no NVLink (intra-node P2P over PCIe off); inter-node = TCP on
#    the 10.5.x cluster net (no IB on dws). The NIC name DIFFERS per node, so
#    _mn_launch.sh pins NCCL_SOCKET_IFNAME by IP subnet (NCCL_IFACE_SUBNET) on
#    each node rather than a hardcoded name. INFO so we can read the topology.
export NCCL_P2P_DISABLE=1
# dws NICs expose RoCE verbs but have no usable IB fabric — NCCL tries RDMA and
# dies on ibv_reg_mr (Cannot allocate memory). Force the pure-TCP socket path.
# On a REAL IB cluster (Helix/bwUniCluster) set NCCL_IB_DISABLE=0 + NCCL_IB_HCA.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IFACE_SUBNET=${NCCL_IFACE_SUBNET:-10.5.}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET
# Make a stuck collective RAISE instead of spinning forever. Note: this does NOT
# shorten the timeout DURATION — that's the init_process_group(timeout=) default
# (~10 min for NCCL); there is no env var for it (NCCL_TIMEOUT is not read by
# NCCL/PyTorch). To fail faster during bring-up, pass a shorter timeout in code.
export TORCH_NCCL_BLOCKING_WAIT=1
export RDF2VEC_TORCHREC_PROGRESS=1

# ── workload (dbpedia2016 corpus already on disk) ──
export CORPUS=${CORPUS:-/ceph/mboeckli/rdf2vec_bench/corpus/dbpedia2016}
export OUT=${OUT:-/ceph/mboeckli/rdf2vec_bench/results/mn_test}
export EPOCHS=${EPOCHS:-1}
export PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-131072}
export LR=${LR:-0.05}
mkdir -p "$OUT"

echo "[mn] nnodes=$SLURM_NNODES master=$MASTER_ADDR:$MASTER_PORT gpus/node=$GPUS_PER_NODE world=$((SLURM_NNODES*GPUS_PER_NODE))"
echo "[mn] nodelist=$SLURM_JOB_NODELIST corpus=$CORPUS epochs=$EPOCHS batch=$PAIR_BATCH_SIZE"

# One torchrun per node (helper resolves per-node SLURM_NODEID + NIC). Env is
# propagated to the srun tasks automatically.
srun --ntasks-per-node=1 bash "$REPO/performance/slurm/_mn_launch.sh"
echo "[mn] srun returned $?"
