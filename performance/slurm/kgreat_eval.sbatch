#!/bin/bash
#SBATCH --job-name=kgreat_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/ceph/mboeckli/rdf2vec_bench/logs/%x_%j.log
# Session-independent KGrEAT evaluation of whatever embeddings are present in
# kg/dbpedia2016/embedding/. Runs the dataset task containers via podman+crun on
# the compute node (verified working). Parametrized by env:
#   REPORT         output table path
#   MARKER         done-marker path (touched ONLY on success)
#   QUALITY_CSV    analyzer csv path
#   DO_THROUGHPUT  "1" to prepend the torchrec batch-size throughput table
#   THROUGHPUT_ARGS  args for analyze_torchrec_throughput.py (when DO_THROUGHPUT=1)
set -uo pipefail

REPO=/ceph/mboeckli/rdf2vecgpu
KG=/ceph/mboeckli/kgreat
KGE=$KG/kg/dbpedia2016/embedding
SMALL=$KG/kg/dbpedia2016/embedding-small
RESULT=$KG/kg/dbpedia2016/result/run_0
EM=$KG/kg/dbpedia2016/entity_mapping.tsv
REPORT="${REPORT:?set REPORT}"
MARKER="${MARKER:?set MARKER}"
QUALITY_CSV="${QUALITY_CSV:-${REPORT%.txt}.csv}"
DO_THROUGHPUT="${DO_THROUGHPUT:-0}"
PYRUN="conda run -n rdf2vecgpu_test python"
rm -f "$MARKER"

source /ceph/mboeckli/miniconda3/etc/profile.d/conda.sh
cd "$REPO"
echo "[eval] host=$(hostname) embeddings:"; ls -la "$KGE"/*.tsv 2>/dev/null

# 1. small (dataset-entity) embeddings so each task loads ~5k rows not ~5M
echo "[eval] building small embeddings"
$PYRUN performance/build_small_embeddings.py "$EM" "$KGE" "$SMALL"

# 2. quarantine any NaN/inf embedding (e.g. a diverged high-LR variant). Fail-SAFE:
#    only move a TSV when the finite-check returns a confirmed positive integer;
#    if the check can't run, keep the TSV (never quarantine on an empty result).
mkdir -p "$KGE/quarantine"
for tsv in "$KGE"/*.tsv; do
  [ -e "$tsv" ] || continue
  bad=$($PYRUN performance/validate_tsv_finite.py "$tsv" 2>/dev/null | tr -dc '0-9')
  if [[ "$bad" =~ ^[0-9]+$ ]] && [ "$bad" -gt 0 ]; then
    echo "[eval] QUARANTINE $(basename "$tsv") (NaN/inf=$bad)"
    mv "$tsv" "$KGE/quarantine/"
    rm -f "$SMALL/$(basename "${tsv%.tsv}")_small.tsv"
  elif [ -z "$bad" ]; then
    echo "[eval] WARN finite-check failed for $(basename "$tsv") -> keeping it"
  else
    echo "[eval] ok $(basename "$tsv") (NaN/inf=$bad)"
  fi
done

# 3. run the 10 GEval tasks on all surviving embeddings
echo "[eval] running KGrEAT task stage"
bash "$KG/run_kgreat_dbpedia2016.sh" task
if ! ls "$RESULT"/*.tsv >/dev/null 2>&1; then
  echo "[eval] FAILED: task stage produced no result TSVs"; exit 1
fi

# 4. quality (+ optional throughput) table. stdout -> REPORT; stderr (conda
#    noise / real errors) -> job log. Marker only on analyzer success.
echo "[eval] building report -> $REPORT"
: > "$REPORT"
if [ "$DO_THROUGHPUT" = "1" ]; then
  echo "================ THROUGHPUT (torchrec pair_batch_size sweep) ================" >> "$REPORT"
  $PYRUN performance/analyze_torchrec_throughput.py ${THROUGHPUT_ARGS} >> "$REPORT"
  echo >> "$REPORT"
  echo "================ QUALITY (GEval, KNOWN eval mode) ================" >> "$REPORT"
fi
if $PYRUN performance/analyze_kgreat_results.py "$RESULT" --mode KNOWN --csv "$QUALITY_CSV" >> "$REPORT"; then
  echo "[eval] DONE"; cat "$REPORT"; touch "$MARKER"
else
  echo "[eval] FAILED: analyzer error"; exit 1
fi
