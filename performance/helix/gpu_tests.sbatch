#!/bin/bash
#SBATCH --job-name=rdf2v_gputests
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.log
# GPU unit tests on a 2-GPU allocation (the torchrec test module needs libcuda
# via fbgemm, so it cannot be collected on a login node). Exit status is the
# pytest status (pipefail), and the pass/fail summary line is never filtered:
# -p no:warnings keeps "N passed, M warnings in" out of the noise pattern.
# Use as the gate for follow-up runs: sbatch --dependency=afterok:<this id> ...
set -euo pipefail
module load devel/python 2>/dev/null || true
module load devel/cuda 2>/dev/null || true
VENV="${VENV:-$HOME/venvs/rdf2vecgpu}"
source "$VENV/bin/activate"
cd /gpfs/bwfor/work/ws/ma_mboeckli-rdf2vec_resource/rdf2vecgpu
NOISE='Warning:|warnings\.warn|LooseVersion|functools\.partial|= partial\(|_get_pg_default|is_fx_tracing|^\s*$'
echo "== node $(hostname -s) gpus=$(nvidia-smi -L | wc -l) =="
rc=0
echo "== single-process: --gpu, not multigpu (+ CPU token_counts tests) =="
python -m pytest -q -rA -p no:cacheprovider -p no:warnings --gpu -m "not multigpu" --tb=short \
  test/embedders/torchrec_word2vec_test.py test/embedders/token_counts_test.py 2>&1 | grep -vE "$NOISE" || rc=1
echo "== multigpu: 2 ranks =="
torchrun --standalone --nproc_per_node=2 -m pytest -q -rA -p no:cacheprovider -p no:warnings --gpu -m multigpu --tb=short \
  test/embedders/torchrec_word2vec_test.py 2>&1 | grep -vE "$NOISE" || rc=1
echo "== done rc=$rc =="
exit $rc
