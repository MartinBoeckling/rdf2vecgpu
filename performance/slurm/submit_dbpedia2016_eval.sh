#!/bin/bash
# dbpedia2016 KGrEAT quality eval: prepare -> {gensim, pytorch, torchrec},
# each training job ALSO exports a KGrEAT entity-embedding TSV straight into
# kg/dbpedia2016/embedding/ so the three backends become directly comparable
# on the GEval downstream tasks (real entity coverage, unlike dbpedia50k).
#
#   gensim/pytorch : sgd @ 0.025      torchrec : rowwiseadagrad @ 0.05
#   (torchrec's fused optimizer-in-backward can't clip -> sgd diverges)
#
# Usage:  bash submit_dbpedia2016_eval.sh
set -euo pipefail

SLURM_DIR=/ceph/mboeckli/rdf2vecgpu/performance/slurm
REPO=/ceph/mboeckli/rdf2vecgpu
WORK=/ceph/mboeckli/rdf2vec_bench
KGE=/ceph/mboeckli/kgreat/kg/dbpedia2016/embedding

# ── shared training hyperparameters (identical across all backends) ──
WALK_STRATEGY=random
WALK_DEPTH=4
WALK_NUMBER=10            # 10 walks/entity over ~4.5M entities -> ~45M walks
EMB_MODEL=skipgram
EPOCHS=5
VECTOR_SIZE=200          # matches the dbpedia50k eval for a like-for-like comparison
WINDOW=5
MIN_COUNT=1
LR=0.025
NEGATIVE=5
RANDOM_STATE=42
CPU_COUNT=20
GRAD_CLIP=0.1
OPTIMIZER=sgd

DATA=$WORK/dbpedia2016.parquet
TAG=dbpedia2016
CORPUS=$WORK/corpus/$TAG
OUTDIR=$WORK/results/$TAG
mkdir -p "$WORK/logs" "$OUTDIR" "$KGE"

if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA not found. Run performance/parse_dbpedia_nt.py first." >&2
  exit 1
fi

# Single-line export string (no embedded whitespace/newlines — sbatch is picky).
EXPORTS="ALL,DATA=$DATA,CORPUS=$CORPUS,OUTDIR=$OUTDIR,TAG=$TAG"
EXPORTS="$EXPORTS,WALK_STRATEGY=$WALK_STRATEGY,WALK_DEPTH=$WALK_DEPTH,WALK_NUMBER=$WALK_NUMBER"
EXPORTS="$EXPORTS,EMB_MODEL=$EMB_MODEL,EPOCHS=$EPOCHS,VECTOR_SIZE=$VECTOR_SIZE,WINDOW=$WINDOW"
EXPORTS="$EXPORTS,MIN_COUNT=$MIN_COUNT,NEGATIVE=$NEGATIVE,RANDOM_STATE=$RANDOM_STATE,CPU_COUNT=$CPU_COUNT,GRAD_CLIP=$GRAD_CLIP"
# LR, OPTIMIZER, EXPORT_KGREAT are appended per-job below.

echo "=== dbpedia2016 KGrEAT eval ==="
echo "  data    : $DATA"
echo "  corpus  : $CORPUS"
echo "  results : $OUTDIR"
echo "  embed   : $KGE/{rdf2vecgpu-gensim,rdf2vecgpu-pytorch,rdf2vecgpu-torchrec-rwa}.tsv"
echo "  walk_number=$WALK_NUMBER depth=$WALK_DEPTH vector_size=$VECTOR_SIZE epochs=$EPOCHS"
echo

PREP=$(sbatch --parsable --job-name="prep_$TAG" \
       --export="$EXPORTS,LR=$LR,OPTIMIZER=$OPTIMIZER,BACKEND=prepare" "$SLURM_DIR/prepare.sbatch")
echo "prepare       -> job $PREP"

declare -A TSV=(
  [gensim]="$KGE/rdf2vecgpu-gensim.tsv"
  [pytorch]="$KGE/rdf2vecgpu-pytorch.tsv"
  [torchrec]="$KGE/rdf2vecgpu-torchrec-rwa.tsv"
)

for B in gensim pytorch torchrec; do
  OPT="$OPTIMIZER"; OLR="$LR"
  if [ "$B" = "torchrec" ]; then OPT="rowwiseadagrad"; OLR="0.05"; fi
  JID=$(sbatch --parsable --dependency=afterok:"$PREP" \
        --job-name="${B}_$TAG" \
        --export="$EXPORTS,LR=$OLR,OPTIMIZER=$OPT,BACKEND=$B,EXPORT_KGREAT=${TSV[$B]}" \
        "$SLURM_DIR/train_${B}.sbatch")
  echo "train $B  -> job $JID  (afterok:$PREP, optimizer=$OPT lr=$OLR, tsv=${TSV[$B]})"
done

echo
echo "watch:   squeue -u \$USER"
echo "results: cat $OUTDIR/*.jsonl ; ls -la $KGE"
