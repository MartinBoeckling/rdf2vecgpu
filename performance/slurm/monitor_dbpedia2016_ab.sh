#!/bin/bash
# A/B: torchrec pair_batch_size sweep on dbpedia2016, fired AFTER the baseline
# run completes (waits for monitor_dbpedia2016.DONE so the baseline TSV + its
# GEval eval are settled and the cluster isn't double-loaded).
#
# Baseline A = the completed run: pair_batch_size 32768, lr 0.05.
# Variants (same prepared corpus, same 5 epochs / dim 200 / window 5):
#     b131k-lr05 : pair_batch_size 131072, lr 0.05   (isolate pure batch effect)
#     b131k-lr10 : pair_batch_size 131072, lr 0.10   (2x LR compensation)
#     b131k-lr20 : pair_batch_size 131072, lr 0.20   (4x linear-scaling)
#
# Then: quarantine any NaN/inf TSV, rebuild small embeddings, re-run the 10 GEval
# tasks on ALL embeddings, and emit a joint throughput + quality report.
#
# Usage:  bash monitor_dbpedia2016_ab.sh
set -uo pipefail

REPO=/ceph/mboeckli/rdf2vecgpu
SLURM_DIR=$REPO/performance/slurm
KG=/ceph/mboeckli/kgreat
KGE=$KG/kg/dbpedia2016/embedding
RESULT=$KG/kg/dbpedia2016/result/run_0
WORK=/ceph/mboeckli/rdf2vec_bench
CORPUS=$WORK/corpus/dbpedia2016
OUTDIR=$WORK/results/dbpedia2016_ab
LOGS=$WORK/logs
STATUS=$LOGS/monitor_dbpedia2016_ab.status
REPORT=$LOGS/dbpedia2016_ab_report.txt
MARKER=$LOGS/monitor_dbpedia2016_ab.DONE
BASE_DONE=$LOGS/monitor_dbpedia2016.DONE
TOTAL_WALKS=50442230
PAIRS_PER_WALK=60                 # skipgram_pairs_per_walk(depth=4, window=5)
BASELINE_JOB=264563              # the completed torchrec run (pbs 32768, lr 0.05)
mkdir -p "$OUTDIR" "$LOGS"; rm -f "$MARKER"; : > "$STATUS"
source /ceph/mboeckli/miniconda3/etc/profile.d/conda.sh

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# ── 0. wait for the baseline run to finish ──
log "waiting for baseline DONE marker ($BASE_DONE)"
while [ ! -f "$BASE_DONE" ]; do sleep 120; done
log "baseline complete; launching pair_batch_size sweep"

# ── shared HP (identical to the baseline eval; only PBS + LR vary) ──
BASE_EXPORTS="ALL,DATA=$WORK/dbpedia2016.parquet,CORPUS=$CORPUS,OUTDIR=$OUTDIR,TAG=dbp2016ab"
BASE_EXPORTS="$BASE_EXPORTS,WALK_STRATEGY=random,WALK_DEPTH=4,WALK_NUMBER=10"
BASE_EXPORTS="$BASE_EXPORTS,EMB_MODEL=skipgram,EPOCHS=5,VECTOR_SIZE=200,WINDOW=5"
BASE_EXPORTS="$BASE_EXPORTS,MIN_COUNT=1,NEGATIVE=5,RANDOM_STATE=42,CPU_COUNT=20"
BASE_EXPORTS="$BASE_EXPORTS,GRAD_CLIP=0.1,OPTIMIZER=rowwiseadagrad,BACKEND=torchrec"

# variant -> "pbs lr"
declare -A VAR=( [b131k-lr05]="131072 0.05" [b131k-lr10]="131072 0.10" [b131k-lr20]="131072 0.20" )
declare -A JID
for v in b131k-lr05 b131k-lr10 b131k-lr20; do
  read pbs lr <<<"${VAR[$v]}"
  tsv="$KGE/rdf2vecgpu-torchrec-$v.tsv"
  jid=$(sbatch --parsable --job-name="torchrec_ab_$v" \
        --export="$BASE_EXPORTS,LR=$lr,PAIR_BATCH_SIZE=$pbs,EXPORT_KGREAT=$tsv,OUT_JSONL=$OUTDIR/torchrec_$v.jsonl" \
        "$SLURM_DIR/train_torchrec.sbatch")
  JID[$v]=$jid
  log "submitted $v (pbs=$pbs lr=$lr) -> job $jid  tsv=$tsv"
done

# ── wait for all variants ──
while :; do
  left=()
  for v in "${!JID[@]}"; do
    st=$(squeue -j "${JID[$v]}" -h -o "%t" 2>/dev/null); [ -n "$st" ] && left+=("$v:$st")
  done
  [ "${#left[@]}" -eq 0 ] && break
  log "variants running: ${left[*]}"; sleep 120
done
log "all variants finished"
for v in b131k-lr05 b131k-lr10 b131k-lr20; do
  log "  $v: $(sacct -j ${JID[$v]} --format=State,Elapsed -n 2>/dev/null | head -1)"
done

# ── quarantine NaN/inf TSVs so KGrEAT doesn't choke (e.g. a diverged high-LR run) ──
mkdir -p "$KGE/quarantine"
for v in b131k-lr05 b131k-lr10 b131k-lr20; do
  tsv="$KGE/rdf2vecgpu-torchrec-$v.tsv"
  [ -f "$tsv" ] || { log "  WARN $v: TSV missing"; continue; }
  bad=$(conda run -n rdf2vecgpu_test python - "$tsv" <<'PY' 2>/dev/null
import sys,pandas as pd,numpy as np
df=pd.read_csv(sys.argv[1],sep="\t",header=None,index_col=0,nrows=200000)
M=df.to_numpy(dtype=float); print(int(np.isnan(M).sum()+np.isinf(M).sum()))
PY
)
  if [ "${bad:-1}" != "0" ]; then
    log "  $v: NaN/inf detected ($bad) -> quarantining (likely diverged)"
    mv "$tsv" "$KGE/quarantine/"
  else
    log "  $v: TSV clean"
  fi
done

# ── rebuild small embeddings (all present TSVs) + re-run tasks on everything ──
log "rebuilding small embeddings"
conda run -n rdf2vecgpu_test python "$REPO/performance/build_small_embeddings.py" \
    "$KG/kg/dbpedia2016/entity_mapping.tsv" "$KGE" "$KG/kg/dbpedia2016/embedding-small" >>"$STATUS" 2>&1
log "re-running KGrEAT task stage on all embeddings"
bash "$KG/run_kgreat_dbpedia2016.sh" task >>"$STATUS" 2>&1

# ── reports ──
log "building throughput + quality reports"
conda run -n rdf2vecgpu_test python "$REPO/performance/analyze_torchrec_throughput.py" \
  "$TOTAL_WALKS" "$PAIRS_PER_WALK" \
  "A-32k-lr05::$LOGS/torchrec_dbpedia2016_${BASELINE_JOB}.log::$BASELINE_JOB::32768" \
  "b131k-lr05::$LOGS/torchrec_ab_b131k-lr05_${JID[b131k-lr05]}.log::${JID[b131k-lr05]}::131072" \
  "b131k-lr10::$LOGS/torchrec_ab_b131k-lr10_${JID[b131k-lr10]}.log::${JID[b131k-lr10]}::131072" \
  "b131k-lr20::$LOGS/torchrec_ab_b131k-lr20_${JID[b131k-lr20]}.log::${JID[b131k-lr20]}::131072" \
  > "$REPORT" 2>>"$STATUS"

{
  echo; echo "================ THROUGHPUT (pair_batch_size sweep) ================"
  cat "$REPORT"
  echo; echo "================ QUALITY (GEval, KNOWN eval mode) ================"
} | tee -a "$STATUS"
conda run -n rdf2vecgpu_test python "$REPO/performance/analyze_kgreat_results.py" \
    "$RESULT" --mode KNOWN --csv "$LOGS/dbpedia2016_ab_quality.csv" 2>>"$STATUS" | tee -a "$REPORT" | tee -a "$STATUS"

log "DONE -> $REPORT"
touch "$MARKER"
