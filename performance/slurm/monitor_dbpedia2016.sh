#!/bin/bash
# Self-driving monitor for the dbpedia2016 KGrEAT eval.
#
#   1. wait for the 3 training jobs (gensim/pytorch/torchrec) to leave the queue
#   2. validate each exported embedding TSV (rows, finite, no NaN)
#   3. trigger the KGrEAT task stage on whatever valid TSVs are present
#   4. run the analysis -> backend-comparison table
#   5. write a report + a DONE marker
#
# Everything is logged to $STATUS. Re-invokable: if results already exist it
# still re-runs the task stage + analysis (KGrEAT overwrites its result dir).
#
# Usage:  bash monitor_dbpedia2016.sh
set -uo pipefail

REPO=/ceph/mboeckli/rdf2vecgpu
KG=/ceph/mboeckli/kgreat
KGE=$KG/kg/dbpedia2016/embedding
RESULT=$KG/kg/dbpedia2016/result/run_0
LOGS=/ceph/mboeckli/rdf2vec_bench/logs
STATUS=$LOGS/monitor_dbpedia2016.status
REPORT=$LOGS/dbpedia2016_comparison.txt
MARKER=$LOGS/monitor_dbpedia2016.DONE
rm -f "$MARKER"

declare -A JOB=( [gensim]=264561 [pytorch]=264562 [torchrec]=264563 )
declare -A TSV=( [gensim]=$KGE/rdf2vecgpu-gensim.tsv \
                 [pytorch]=$KGE/rdf2vecgpu-pytorch.tsv \
                 [torchrec]=$KGE/rdf2vecgpu-torchrec-rwa.tsv )

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

source /ceph/mboeckli/miniconda3/etc/profile.d/conda.sh

: > "$STATUS"
log "monitor started; waiting for jobs ${JOB[*]}"

# ── 1. wait for all 3 training jobs to finish ──
while :; do
  left=()
  for b in gensim pytorch torchrec; do
    st=$(squeue -j "${JOB[$b]}" -h -o "%t" 2>/dev/null)
    [ -n "$st" ] && left+=("$b:$st")
  done
  [ "${#left[@]}" -eq 0 ] && break
  log "still running: ${left[*]}"
  sleep 120
done
log "all 3 training jobs left the queue"

# ── 2. report final state + validate TSVs ──
PRESENT=()
for b in gensim pytorch torchrec; do
  state=$(sacct -j "${JOB[$b]}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
  if [ -f "${TSV[$b]}" ]; then
    chk=$(conda run -n rdf2vecgpu_test python - "${TSV[$b]}" <<'PY' 2>/dev/null
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], sep="\t", header=None, index_col=0)
M = df.to_numpy(dtype=float)
nan = int(np.isnan(M).sum() + np.isinf(M).sum())
print(f"rows={len(df)} dim={M.shape[1]} nan_inf={nan} l1={np.abs(M[0]).sum():.4f}")
PY
)
    log "$b: state=$state  TSV ok -> $chk"
    PRESENT+=("$b")
  else
    log "$b: state=$state  TSV MISSING (${TSV[$b]})"
  fi
done

if [ "${#PRESENT[@]}" -eq 0 ]; then
  log "FATAL: no embedding TSVs produced; aborting before task stage"
  touch "$MARKER"; exit 1
fi
log "embeddings present for: ${PRESENT[*]}"

# ── 2b. small (dataset-entity) embeddings so tasks load ~5k rows, not ~5M ──
log "building small embeddings (filter full TSVs to gold entities)"
conda run -n rdf2vecgpu_test python "$REPO/performance/build_small_embeddings.py" \
    "$KG/kg/dbpedia2016/entity_mapping.tsv" "$KGE" \
    "$KG/kg/dbpedia2016/embedding-small" >>"$STATUS" 2>&1
log "small embeddings:"; ls -la "$KG/kg/dbpedia2016/embedding-small" 2>/dev/null | tee -a "$STATUS"

# ── 3. KGrEAT task stage (runs on all kg/dbpedia2016/embedding/*.tsv) ──
bash "$KG/run_kgreat_dbpedia2016.sh" task >>"$STATUS" 2>&1
log "task stage finished; result dir: $RESULT"
ls -la "$RESULT" 2>/dev/null | tee -a "$STATUS"

# ── 4. analysis ──
log "running analysis"
conda run -n rdf2vecgpu_test python "$REPO/performance/analyze_kgreat_results.py" \
    "$RESULT" --mode KNOWN --csv "$LOGS/dbpedia2016_comparison.csv" > "$REPORT" 2>>"$STATUS"
{
  echo "================ dbpedia2016 KGrEAT comparison (KNOWN eval mode) ================"
  cat "$REPORT"
} | tee -a "$STATUS"

# ── 5. done ──
log "DONE -> report: $REPORT  csv: $LOGS/dbpedia2016_comparison.csv"
touch "$MARKER"
