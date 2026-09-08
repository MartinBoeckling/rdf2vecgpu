#!/bin/bash
#SBATCH --job-name=torchrec_gpu8
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=%x_%j.log
# Single-node torchrec training on a Helix gpu8 node (8x A100/80GB = 640 GB
# VRAM, or 8x H200/141GB = 1128 GB; 2.2 TB RAM; gpu-single allows 120 h).
# Header defaults are the SMOKE shape (2x A100, dbpedia2016, 1 epoch). Scale up
# at submit time — full-Wikidata d100 fits ONE A100-gpu8 node (314 GB tables ->
# ~39 GB/GPU), d200 needs the H200 flavor (629 GB -> ~79 GB/GPU):
#   sbatch --gres=gpu:A100:8 --cpus-per-task=48 --mem=800G --time=96:00:00 \
#     --export=ALL,GPUS=8,CORPUS=$WS/corpus/wikidata_v100,EPOCHS=5,... this-file
# Single node => torchrun --standalone, NO multi-node rendezvous, and the
# all-to-all stays on NVLink/NVSwitch. Do NOT set NCCL_P2P_DISABLE here (that
# flag is the dws-A6000 PCIe workaround; on NVLink boxes it disables the fast
# path).
set -euo pipefail

module load devel/python 2>/dev/null || true   # venv needs libpython3.13.so.1.0
module load devel/cuda 2>/dev/null || true
VENV="${VENV:-$HOME/venvs/rdf2vecgpu}"
source "$VENV/bin/activate"
REPO="/gpfs/bwfor/work/ws/ma_mboeckli-rdf2vec_resource/rdf2vecgpu"
cd "$REPO"

export GPUS=${GPUS:-2}

WS="${WS:-$(ws_find rdf2vec_resource 2>/dev/null || echo "$HOME")}"

# ── RACE_GROUP: "whoever gets past start-up first wins", no external watcher ──
# Queue the same run on two GPU types (job names sharing this prefix). A job
# claims the race only once its torchrun has printed the first progress line
# (model built, sharded, step 0 done — the phase that OOM-killed 14593196 six
# minutes in, after it had already cancelled its A100 sibling); only then does
# it cancel every PENDING sibling. Until then siblings stay queued, so a
# start-up failure hands the slot to the other GPU type instead of emptying the
# queue. The claim is an atomic winner file under $WS/results; a job that
# starts while another RUNNING job holds it cancels itself, a claim held by a
# job that is no longer running is stale and gets replaced.
WIN=""
race_sibling_running() {   # $1 = job id read from the winner file
  [ -n "$1" ] && [ "$1" != "$SLURM_JOB_ID" ] \
    && squeue -h -j "$1" -o %T 2>/dev/null | grep -q RUNNING
}
if [ -n "${RACE_GROUP:-}" ]; then
  WIN="$WS/results/.race_${RACE_GROUP}.winner"
  w=$(cat "$WIN" 2>/dev/null || true)
  if race_sibling_running "$w"; then
    echo "[gpu8] race: sibling $w already won $RACE_GROUP — cancelling self"
    scancel "$SLURM_JOB_ID"; sleep 60; exit 0
  fi
  [ -n "$w" ] && { echo "[gpu8] race: stale claim by $w (not running) — removing"; rm -f "$WIN"; }
fi
export RDF2VEC_TORCHREC_PROGRESS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── workload ──
export CORPUS=${CORPUS:-$WS/corpus/dbpedia2016}
export OUT=${OUT:-$WS/results/gpu8}
export EPOCHS=${EPOCHS:-1}
export PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-131072}
export LR=${LR:-0.10}
# DCP checkpoints every CHECKPOINT_INTERVAL_MIN into $OUT/ckpt (auto-resume on
# resubmit). CHECKPOINT_DIR="" disables. A full-Wikidata d100 checkpoint is
# ~300 GB (tables + Adagrad state) written shard-parallel by all ranks.
export CHECKPOINT_DIR=${CHECKPOINT_DIR-$OUT/ckpt}
export CHECKPOINT_INTERVAL_MIN=${CHECKPOINT_INTERVAL_MIN:-120}
mkdir -p "$OUT"

echo "[gpu8] host=$(hostname -s) gpus=$GPUS corpus=$CORPUS epochs=$EPOCHS batch=$PAIR_BATCH_SIZE lr=$LR ckpt=${CHECKPOINT_DIR:-off}"
[ -f "$CORPUS/token_counts.npy" ] || echo "[gpu8] WARN: $CORPUS/token_counts.npy missing — every rank will count tokens before step 0 (see performance/helix/token_counts.sbatch)"
nvidia-smi topo -m | sed -n "1,$((GPUS+2))p"

# RUN_TESTS=1: run the GPU unit tests on this allocation before training (the
# torchrec test module needs libcuda via fbgemm, so it cannot run on a login
# node). Failures are reported loudly and the job still trains — one GPU
# allocation then yields both signals — but the job exits 1 at the end.
# The noise filter must never match the "N passed, M warnings in" summary
# line (it once did, via "warn") — hence -p no:warnings and the narrower pattern.
TESTS_FAILED=0
if [ "${RUN_TESTS:-0}" = "1" ]; then
  echo "[gpu8] GPU unit tests (single-process)"
  if ! python -m pytest -q -rA -p no:cacheprovider -p no:warnings --gpu -m "not multigpu" --tb=short \
       test/embedders/torchrec_word2vec_test.py test/embedders/token_counts_test.py 2>&1 \
       | grep -vE "Warning:|warnings\.warn|LooseVersion|^$" | tail -40; then
    TESTS_FAILED=1; echo "[gpu8] !! single-process GPU tests FAILED"
  fi
  echo "[gpu8] GPU unit tests (multigpu, 2 ranks)"
  if ! torchrun --standalone --nproc_per_node=2 -m pytest -q -rA -p no:cacheprovider -p no:warnings --gpu -m multigpu --tb=short \
       test/embedders/torchrec_word2vec_test.py 2>&1 \
       | grep -vE "Warning:|warnings\.warn|LooseVersion|functools.partial|= partial\(|_get_pg_default|is_fx_tracing|^$" | tail -60; then
    TESTS_FAILED=1; echo "[gpu8] !! multigpu tests FAILED"
  fi
  echo "[gpu8] tests done (failed=$TESTS_FAILED)"
fi

LOG="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"   # this job's --output
LOG_START=$(wc -l < "$LOG" 2>/dev/null || echo 0)   # test output above may contain progress lines
torchrun --standalone --nproc_per_node="$GPUS" \
  performance/bench_backends.py --backend torchrec \
    --corpus "$CORPUS" --out "$OUT/run.jsonl" --tag helix_gpu8 \
    --walk-depth 4 --walk-number 10 --vector-size "${VECTOR_SIZE:-200}" --window-size 5 \
    --epochs "$EPOCHS" --negative-samples 5 --min-count 1 \
    --learning-rate "$LR" --optimizer rowwiseadagrad --pair-batch-size "$PAIR_BATCH_SIZE" \
    ${CHECKPOINT_DIR:+--checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-interval-minutes "$CHECKPOINT_INTERVAL_MIN"} \
    ${EXPORT_KGREAT:+--export-kgreat "$EXPORT_KGREAT"} &
TR_PID=$!

# Race claim: poll this job's own log until torchrun prints its first progress
# line, then claim atomically and cancel the PENDING siblings.
if [ -n "${RACE_GROUP:-}" ]; then
  while kill -0 "$TR_PID" 2>/dev/null; do
    if tail -n +"$((LOG_START + 1))" "$LOG" 2>/dev/null | grep -q "\[torchrec\] epoch"; then
      if ( set -o noclobber; echo "$SLURM_JOB_ID" > "$WIN" ) 2>/dev/null; then
        echo "[gpu8] race: past start-up — claimed $RACE_GROUP, cancelling pending siblings"
        while read -r jid jname jstate; do
          [ "$jid" = "$SLURM_JOB_ID" ] && continue
          case "$jname" in "$RACE_GROUP"*) ;; *) continue ;; esac
          echo "[gpu8] race: cancelling sibling $jid ($jname, $jstate)"; scancel "$jid" || true
        done < <(squeue -u "$USER" -h -o "%i %j %T")
        break
      fi
      w=$(cat "$WIN" 2>/dev/null || true)
      if race_sibling_running "$w"; then
        echo "[gpu8] race: sibling $w claimed first — stopping torchrun and cancelling self"
        kill "$TR_PID" 2>/dev/null || true; sleep 20; scancel "$SLURM_JOB_ID"; sleep 60; exit 0
      fi
      rm -f "$WIN"   # stale claim (holder not running) — retry on the next poll
    fi
    sleep 30
  done
fi

rc=0; wait "$TR_PID" || rc=$?
echo "[gpu8] torchrun returned $rc"
if [ -n "$WIN" ] && [ "$rc" -ne 0 ] && [ "$(cat "$WIN" 2>/dev/null || true)" = "$SLURM_JOB_ID" ]; then
  rm -f "$WIN"   # release the claim so a resubmitted sibling can take the race
fi
[ "$TESTS_FAILED" = 0 ] || { echo "[gpu8] exiting 1: unit tests failed (see above)"; exit 1; }
exit "$rc"
