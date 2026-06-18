#!/bin/bash
#SBATCH --job-name=rdf2v_torchrec
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# torchrec sharded embeddings (4 ranks under torchrun). A6000 has no NVLink, so
# disable NCCL P2P over PCIe. Training-only timing from the shared corpus; only
# rank 0 writes the result row.

source /ceph/mboeckli/rdf2vecgpu/performance/slurm/env.sh

export NCCL_P2P_DISABLE=1

torchrun --standalone --nproc_per_node=4 \
  performance/bench_backends.py --backend torchrec \
  --corpus "$CORPUS" --out "$OUTDIR/torchrec.jsonl" --tag "$TAG" \
  "${HP_ARGS[@]}"
