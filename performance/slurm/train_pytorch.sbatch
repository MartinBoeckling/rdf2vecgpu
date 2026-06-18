#!/bin/bash
#SBATCH --job-name=rdf2v_pytorch
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# pytorch single-GPU. GPU-resident in-memory feed (cuDF -> DLPack), no temp
# parquet. Training-only timing from the shared corpus.

source /ceph/mboeckli/rdf2vecgpu/performance/slurm/env.sh

python performance/bench_backends.py --backend pytorch \
  --corpus "$CORPUS" --out "$OUTDIR/pytorch.jsonl" --tag "$TAG" \
  "${HP_ARGS[@]}"
