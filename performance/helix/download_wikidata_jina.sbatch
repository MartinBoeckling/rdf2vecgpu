#!/bin/bash
#SBATCH --job-name=dl_wd_jina
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
# Download the Jina embeddings for Wikidata entities from the HF Hub onto the
# workspace. CPU-only, network-bound: the "en" split is 227 parquet shards /
# ~48 GB, so this is a download job, not a compute job.
#
# The transfer is resumable — re-submitting the same script skips shards that
# are already complete, so a walltime overrun is not fatal.
#
# Overrides at submit time:
#   sbatch --export=ALL,LANGUAGE=de performance/helix/download_wikidata_jina.sbatch
#   sbatch --export=ALL,DEST=/some/other/dir ... (default: $WS/data/wikidata_jina_embeddings)
set -euo pipefail

module load devel/python 2>/dev/null || true   # venv needs libpython3.13.so.1.0
VENV="${VENV:-$HOME/venvs/rdf2vecgpu}"
source "$VENV/bin/activate"

WS="${WS:-$(ws_find rdf2vec_resource 2>/dev/null || echo "$HOME")}"
export LANGUAGE="${LANGUAGE:-en}"
export DEST="${DEST:-$WS/data/wikidata_jina_embeddings}"

# Never let the HF cache land in $HOME (small quota) — keep it on the workspace.
# (hf_xet is installed and gives the fast transfer path; hf_transfer is not,
# so do NOT set HF_HUB_ENABLE_HF_TRANSFER — hub 1.30 errors out without it.)
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"
mkdir -p "$DEST" "$HF_HOME"

echo "[dl] host=$(hostname -s) job=${SLURM_JOB_ID:-NA} lang=$LANGUAGE"
echo "[dl] dest=$DEST hf_home=$HF_HOME"

# Compute nodes may sit behind a firewall; fail fast and loud rather than
# burning walltime on retries.
if ! curl -sS -m 30 -o /dev/null https://huggingface.co/api/models; then
  echo "[dl] ERROR: no route to huggingface.co from this compute node." >&2
  echo "[dl] Run the same download on a login node instead, or ask support" >&2
  echo "[dl] about a proxy (http_proxy/https_proxy) for the cpu-single nodes." >&2
  exit 1
fi

python - <<'PY'
import os
from huggingface_hub import snapshot_download

REPO = "philippesaade/Wikidata_Vectors_0.2"
lang = os.environ["LANGUAGE"]
dest = os.environ["DEST"]

path = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=f"data/{lang}/*.parquet",
    local_dir=dest,          # already-complete shards are skipped on re-run
    max_workers=8,
)
print(f"[dl] snapshot at {path}")

shards = sorted((p for p in os.scandir(os.path.join(dest, "data", lang))
                 if p.name.endswith(".parquet")), key=lambda p: p.name)
total = sum(p.stat().st_size for p in shards)
print(f"[dl] {len(shards)} parquet shards, {total / 1e9:.2f} GB")

# Cheap integrity check: the last shard must be a readable parquet file.
import pyarrow.parquet as pq
pf = pq.ParquetFile(shards[-1].path)
print(f"[dl] {shards[-1].name}: {pf.metadata.num_rows} rows, "
      f"columns={pf.schema_arrow.names}")
PY

echo "[dl] done. Read it back with:"
echo "  load_dataset('parquet', data_files='$DEST/data/$LANGUAGE/*.parquet')"
