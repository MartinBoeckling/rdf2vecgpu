#!/bin/bash
#SBATCH --job-name=rdf2v_gensim
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# gensim (CPU training). Still needs 1 GPU: fit_from_walks loads the vocab and
# token counts via cudf. Training-only timing from the shared corpus.

source /ceph/mboeckli/rdf2vecgpu/performance/slurm/env.sh

# Optional KGrEAT entity-embedding export (set EXPORT_KGREAT to a .tsv path).
EXPORT_ARG=()
[ -n "${EXPORT_KGREAT:-}" ] && EXPORT_ARG+=(--export-kgreat "$EXPORT_KGREAT")

python performance/bench_backends.py --backend gensim \
  --corpus "$CORPUS" --out "$OUTDIR/gensim.jsonl" --tag "$TAG" \
  "${HP_ARGS[@]}" ${EXPORT_ARG[@]+"${EXPORT_ARG[@]}"}
