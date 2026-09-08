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
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    walks = os.path.join(a.corpus, "walks")
    vocab = os.path.join(a.corpus, "vocab.parquet")
    vocab_size = pq.ParquetFile(vocab).metadata.num_rows
    out = token_counts_cache_path(vocab)
    print(f"[token_counts] corpus={a.corpus} vocab_size={vocab_size:,} workers={a.workers} -> {out}", flush=True)

    t0 = time.perf_counter()
    counts = token_counts_from_walk_parquet(walks, vocab_size, workers=a.workers)
    dt = time.perf_counter() - t0
    save_token_counts_cache(out, counts)

    nz = int((counts > 0).sum())
    print(f"[token_counts] done in {dt/60:.1f} min: total={int(counts.sum()):,} "
          f"nonzero={nz:,}/{vocab_size:,} ({nz/vocab_size*100:.1f}% of vocab touched) "
          f"max={int(counts.max()):,} -> {out} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
