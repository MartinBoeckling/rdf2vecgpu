#!/bin/bash
#SBATCH --job-name=rdf2v_torchrec
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1-12:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# torchrec sharded embeddings (4 ranks under torchrun). A6000 has no NVLink, so
# disable NCCL P2P over PCIe. Training-only timing from the shared corpus; only
# rank 0 writes the result row.

source /ceph/mboeckli/rdf2vecgpu/performance/slurm/env.sh

export NCCL_P2P_DISABLE=1
# torchrec's Fabric loop has no Lightning progress bar; this prints a flushed
# rank-0 progress line (epoch | step | loss | steps/s) every 200 steps so the
# otherwise-buffered job log shows live progress.
export RDF2VEC_TORCHREC_PROGRESS=1

# Optional KGrEAT entity-embedding export (set EXPORT_KGREAT to a .tsv path).
# Each rank exports its shard; rank 0 concatenates into the final TSV.
EXPORT_ARG=()
[ -n "${EXPORT_KGREAT:-}" ] && EXPORT_ARG+=(--export-kgreat "$EXPORT_KGREAT")

# Optional pair_batch_size override (batch-size sweep). Unset -> config default.
PBS_ARG=()
[ -n "${PAIR_BATCH_SIZE:-}" ] && PBS_ARG+=(--pair-batch-size "$PAIR_BATCH_SIZE")

# Optional distinct metrics file (so sweep variants don't share one jsonl).
OUT_JSONL="${OUT_JSONL:-$OUTDIR/torchrec.jsonl}"

torchrun --standalone --nproc_per_node=4 \
  performance/bench_backends.py --backend torchrec \
  --corpus "$CORPUS" --out "$OUT_JSONL" --tag "$TAG" \
  "${HP_ARGS[@]}" ${EXPORT_ARG[@]+"${EXPORT_ARG[@]}"} ${PBS_ARG[@]+"${PBS_ARG[@]}"}
