#!/bin/bash
# One-shot environment setup for rdf2vecgpu torchrec training on Helix.
# Run from the repo root on a Helix login node (or inside an interactive job):
#   bash performance/helix/setup_env.sh [VENV_DIR]
# Pins mirror the known-good dws env (rdf2vecgpu_test, verified 2026-09):
#   torch 2.9.1 / torchrec 1.4.0 / fbgemm-gpu 1.4.1 (cu128), lightning 2.5.6,
#   cudf/cugraph 25.6 (top-level imports in gpu_rdf2vec.py need them present,
#   even though fit_from_walks never touches cuGraph).
set -euo pipefail

VENV="${1:-$HOME/venvs/rdf2vecgpu}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

module load devel/cuda 2>/dev/null || echo "WARN: devel/cuda module not found — relying on wheel-bundled CUDA libs (usually fine for cu128 wheels)"

# Python >= 3.12. Prefer a python module if present, else system python3.
PY=python3
module load devel/python 2>/dev/null && PY=python3 || true
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || { echo "ERROR: need python >= 3.12 (module avail devel/python)"; exit 1; }

"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# torch + torchrec + fbgemm from the cu128 index; the rest from PyPI.
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1
pip install torchrec==1.4.0 fbgemm-gpu==1.4.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install \
    "cudf-cu12==25.6.0" "cugraph-cu12==25.6.*" "dask-cudf-cu12==25.6.*" \
    "dask[complete]" "lightning==2.5.6" "loguru==0.7.3" "numpy==2.2.6" \
    "pydantic==2.12.4" "pyarrow==19.0.1" "rdflib" "tables" "tqdm" "pandas"

pip install -e "$REPO" --no-deps

python - <<'EOF'
import torch, torchrec, lightning, pyarrow
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available())
import rdf2vecgpu
print("rdf2vecgpu import OK:", rdf2vecgpu.__file__)
EOF
echo "env ready: source $VENV/bin/activate"
