#!/bin/bash
#SBATCH --job-name=rdf2v_pytorch_mg
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# pytorch multi-GPU (Lightning DDP, 4 ranks). Launched as plain python — Lightning
# spawns the workers itself (NOT torchrun). A6000 has no NVLink, so NCCL P2P over
# PCIe deadlocks -> disable it. Training-only timing from the shared corpus.

source /ceph/mboeckli/rdf2vecgpu/performance/slurm/env.sh

export NCCL_P2P_DISABLE=1

python performance/bench_backends.py --backend pytorch-mg \
  --corpus "$CORPUS" --out "$OUTDIR/pytorch-mg.jsonl" --tag "$TAG" \
  "${HP_ARGS[@]}"
