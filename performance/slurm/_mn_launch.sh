#!/bin/bash
# Per-node torchrun launcher for multi-node torchrec, invoked once per node by
# `srun --ntasks-per-node=1`. Reads workload + rendezvous config from the env
# exported by train_torchrec_multinode.sbatch.
#
# Key trick: the internal-cluster NIC has DIFFERENT names on different nodes
# (e.g. enp195s0f0 vs ens8f0), so we pin NCCL by IP SUBNET, not by a hardcoded
# interface name. Override the subnet via NCCL_IFACE_SUBNET (default 10.5. for
# dws); falls back to bond0.
set -uo pipefail
cd "$REPO"

SUB="${NCCL_IFACE_SUBNET:-10.5.}"
IF=$(ip -o -4 addr show | awk -v s="$SUB" 'index($4, s)==1 {print $2; exit}')
[ -z "$IF" ] && IF=bond0
export NCCL_SOCKET_IFNAME="$IF"

echo "[mn] node_rank=${SLURM_NODEID}/${SLURM_NNODES} host=$(hostname -s) iface=${IF} CVD=${CUDA_VISIBLE_DEVICES:-?}"

exec torchrun \
  --nnodes="$SLURM_NNODES" \
  --node_rank="$SLURM_NODEID" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  performance/bench_backends.py --backend torchrec \
    --corpus "$CORPUS" --out "$OUT/mn.jsonl" --tag mn_test \
    --walk-depth 4 --walk-number 10 --vector-size 200 --window-size 5 \
    --epochs "$EPOCHS" --negative-samples 5 --min-count 1 \
    --learning-rate "$LR" --optimizer rowwiseadagrad --pair-batch-size "$PAIR_BATCH_SIZE" \
    ${EXPORT_KGREAT:+--export-kgreat "$EXPORT_KGREAT"}
