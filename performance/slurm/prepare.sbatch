#!/bin/bash
#SBATCH --job-name=rdf2v_prep
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# Prepare the shared walk+vocab corpus ONCE (single-GPU). All training jobs
# depend (afterok) on this and read $CORPUS/{walks,vocab.parquet}.

source /ceph/mboeckli/rdf2vecgpu/performance/slurm/env.sh

echo "[prepare] data=$DATA -> corpus=$CORPUS (walk_number=$WALK_NUMBER depth=$WALK_DEPTH)"
python performance/bench_backends.py --prepare \
  --data "$DATA" --corpus "$CORPUS" \
  --out "$OUTDIR/prepare.jsonl" --tag "$TAG" \
  "${HP_ARGS[@]}"

echo "[prepare] done. corpus contents:"
ls -la "$CORPUS" || true
