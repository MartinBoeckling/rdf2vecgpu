#!/usr/bin/env python
"""Precompute the walk-corpus token counts once and cache them next to the vocab.

    python performance/helix/make_token_counts.py $CORPUS [--workers N]

Writes $CORPUS/token_counts.npy (int64, one entry per vocab token), which
GPU_RDF2Vec._load_corpus_vocab picks up instead of re-streaming the whole walks
parquet on every torchrun rank before step 0 (~10 h per rank at Wikidata scale).
CPU-only: pyarrow + numpy, parallel over shards.
"""
import argparse
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from rdf2vecgpu.embedders.word2vec_loader import (  # noqa: E402
    token_counts_from_walk_parquet, token_counts_cache_path, save_token_counts_cache,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="prepared corpus dir holding walks/ and vocab.parquet")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes; 0 = fit the job's memory: each worker holds one "
                         "8 B x vocab accumulator plus ~1.5 GB of chunk working set")
    ap.add_argument("--rows-per-chunk", type=int, default=4_000_000)
    a = ap.parse_args()

    walks = os.path.join(a.corpus, "walks")
    vocab = os.path.join(a.corpus, "vocab.parquet")
    vocab_size = pq.ParquetFile(vocab).metadata.num_rows
    if a.workers <= 0:
        # SLURM_MEM_PER_NODE is in MB; outside Slurm fall back to the CPU count.
        mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0") or 0)
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0) or max(1, (os.cpu_count() or 2) - 1)
        per_worker_gb = 8 * vocab_size / 1e9 + 1.5
        a.workers = max(1, min(cpus, int((mem_mb / 1024 - 8) / per_worker_gb))) if mem_mb else max(1, cpus)
    out = token_counts_cache_path(vocab)
    print(f"[token_counts] corpus={a.corpus} vocab_size={vocab_size:,} workers={a.workers} -> {out}", flush=True)

    t0 = time.perf_counter()
    counts = token_counts_from_walk_parquet(walks, vocab_size, workers=a.workers, rows_per_chunk=a.rows_per_chunk)
    dt = time.perf_counter() - t0
    save_token_counts_cache(out, counts)

    nz = int((counts > 0).sum())
    print(f"[token_counts] done in {dt/60:.1f} min: total={int(counts.sum()):,} "
          f"nonzero={nz:,}/{vocab_size:,} ({nz/vocab_size*100:.1f}% of vocab touched) "
          f"max={int(counts.max()):,} -> {out} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
