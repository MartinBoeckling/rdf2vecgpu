#!/bin/bash
# Submit one prepare -> {gensim, pytorch, pytorch-mg, torchrec} dependency chain
# for a dataset. The four training jobs all wait on `afterok` of the single
# prepare job, then train from the SAME corpus (training-only timing).
#
# Usage:
#   bash submit_sweep.sh fb15k237
#   bash submit_sweep.sh wikidata5m
set -euo pipefail

DS="${1:?usage: submit_sweep.sh <fb15k237|wikidata5m>}"
SLURM_DIR=/ceph/mboeckli/rdf2vecgpu/performance/slurm
REPO=/ceph/mboeckli/rdf2vecgpu
WORK=/ceph/mboeckli/rdf2vec_bench

# ── shared training hyperparameters (identical across all backends) ──
WALK_STRATEGY=random
WALK_DEPTH=4
EMB_MODEL=skipgram
EPOCHS=5
VECTOR_SIZE=100
WINDOW=5
MIN_COUNT=1
LR=0.025
NEGATIVE=5
RANDOM_STATE=42
CPU_COUNT=20
GRAD_CLIP=0.1
OPTIMIZER=sgd

# ── per-dataset settings ──
case "$DS" in
  fb15k237)
    DATA=$REPO/data/fb15k-237/Release/fb15k_kg.parquet
    TAG=fb15k237
    WALK_NUMBER=100
    ;;
  wikidata5m)
    DATA=$REPO/data/wikidata5m/wikidata5m_kg.parquet
    TAG=wikidata5m_wn20      # walk_number=20 keeps vertex-leaves < cuDF 2.1B int32 cap
    WALK_NUMBER=20
    ;;
  *)
    echo "unknown dataset '$DS' (expected fb15k237 | wikidata5m)" >&2
    exit 1
    ;;
esac

CORPUS=$WORK/corpus/$TAG
OUTDIR=$WORK/results/$TAG
mkdir -p "$WORK/logs" "$OUTDIR"

# Single-line export string (no embedded whitespace/newlines — sbatch is picky).
EXPORTS="ALL,DATA=$DATA,CORPUS=$CORPUS,OUTDIR=$OUTDIR,TAG=$TAG"
EXPORTS="$EXPORTS,WALK_STRATEGY=$WALK_STRATEGY,WALK_DEPTH=$WALK_DEPTH,WALK_NUMBER=$WALK_NUMBER"
EXPORTS="$EXPORTS,EMB_MODEL=$EMB_MODEL,EPOCHS=$EPOCHS,VECTOR_SIZE=$VECTOR_SIZE,WINDOW=$WINDOW"
EXPORTS="$EXPORTS,MIN_COUNT=$MIN_COUNT,NEGATIVE=$NEGATIVE,RANDOM_STATE=$RANDOM_STATE,CPU_COUNT=$CPU_COUNT,GRAD_CLIP=$GRAD_CLIP"
# LR and OPTIMIZER are appended per-job below (torchrec needs rowwiseadagrad +
# a GloVe-scale LR; the others use the dataset default LR + sgd).

echo "=== sweep: $TAG ==="
echo "  data   : $DATA"
echo "  corpus : $CORPUS"
echo "  results: $OUTDIR"
echo "  walk_number=$WALK_NUMBER depth=$WALK_DEPTH vector_size=$VECTOR_SIZE epochs=$EPOCHS"
echo

PREP=$(sbatch --parsable --job-name="prep_$TAG" \
       --export="$EXPORTS,LR=$LR,OPTIMIZER=$OPTIMIZER,BACKEND=prepare" "$SLURM_DIR/prepare.sbatch")
echo "prepare      -> job $PREP"

for B in gensim pytorch pytorch_mg torchrec; do
  # torchrec's fused optimizer-in-backward can't clip gradients, so sgd diverges;
  # rowwiseadagrad (GloVe's per-row Adagrad) is stable + recovers structure at a
  # GloVe-scale LR. gensim/pytorch keep sgd at the dataset default LR.
  OPT="$OPTIMIZER"; OLR="$LR"
  if [ "$B" = "torchrec" ]; then OPT="rowwiseadagrad"; OLR="0.05"; fi
  JID=$(sbatch --parsable --dependency=afterok:"$PREP" \
        --job-name="${B}_$TAG" \
        --export="$EXPORTS,LR=$OLR,OPTIMIZER=$OPT,BACKEND=$B" "$SLURM_DIR/train_${B}.sbatch")
  echo "train $B  -> job $JID  (afterok:$PREP, optimizer=$OPT lr=$OLR)"
done

echo
echo "watch:   squeue -u \$USER"
echo "results: cat $OUTDIR/*.jsonl"
