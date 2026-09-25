"""Host-memory probe for the torchrec pair loader — no GPU, no torchrec.

Runs exactly the loader stack one training rank runs
(``_WalkSeqIterable`` → ``_PairsFromWalks`` → ``_ThreadPrefetch`` → DataLoader,
see ``GPU_RDF2Vec._build_walk_pair_loader``) over a walks directory and logs,
every ``--log-every`` batches, this process's resident memory and pyarrow's
pool statistics. Written for 2026-09-21 job 14697729: eight ranks reached the
1500G cgroup ceiling after one full-Wikidata epoch where the H200 run two
weeks earlier peaked at 1.12 TB, and every rank streams all 304 GB of walks
per epoch — the loader is the first suspect for a slow climb.

A flat RSS after the first few thousand batches clears the loader; a slope
gives the leak rate per batch, and the mitigation flags test the fixes in
the same job:

  --release-every N   call ``pa.default_memory_pool().release_unused()``
                      every N batches (pool arenas back to the OS)
  ARROW_DEFAULT_MEMORY_POOL=system|jemalloc|mimalloc   (env, read by pyarrow
                      at import) — compare allocators

  python performance/helix/probe_loader_memory.py \\
      --walks $WS/corpus/wikidata_v100/walks --world-size 8 --rank 0 \\
      --max-batches 200000 --log-every 2000
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _host_mem_gib() -> tuple[float, float]:
    cur = peak = float("nan")
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 2**20
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 2**20
    return cur, peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walks", required=True, help="walks parquet directory (corpus/<name>/walks)")
    ap.add_argument("--world-size", type=int, default=8, help="ranks the real run has (row stride)")
    ap.add_argument("--rank", type=int, default=0, help="which rank's stride to keep")
    ap.add_argument("--pair-batch-size", type=int, default=1_048_576)
    ap.add_argument("--walk-depth", type=int, default=4)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--read-chunk-size", type=int, default=262_144)
    ap.add_argument("--prefetch-depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1, help="re-iterate the loader this many times")
    ap.add_argument("--max-batches", type=int, default=0, help="stop after this many batches per epoch (0 = whole epoch)")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--release-every", type=int, default=0, help="release_unused() on the pyarrow pool every N batches")
    ap.add_argument("--no-prefetch", action="store_true", help="skip the _ThreadPrefetch stage")
    args = ap.parse_args()

    import pyarrow as pa
    import torch
    from torch.utils.data import DataLoader

    from rdf2vecgpu.embedders.word2vec_loader import (
        _PairsFromWalks,
        _ThreadPrefetch,
        _WalkSeqIterable,
        walks_per_pair_batch,
    )

    pool = pa.default_memory_pool()
    print(
        f"[probe] pid={os.getpid()} host={os.uname().nodename} pyarrow={pa.__version__} "
        f"pool={pool.backend_name} torch={torch.__version__} walks={args.walks} "
        f"world_size={args.world_size} rank={args.rank} pair_batch={args.pair_batch_size} "
        f"release_every={args.release_every} prefetch={'off' if args.no_prefetch else args.prefetch_depth}",
        flush=True,
    )
    rss0, _ = _host_mem_gib()
    print(f"[probe] baseline before loader build: rss {rss0:.2f} GiB", flush=True)

    ds = _WalkSeqIterable(
        args.walks,
        emit_batch_size=walks_per_pair_batch(args.pair_batch_size, args.walk_depth, args.window),
        read_chunk_size=args.read_chunk_size,
        shuffle=True,
        seed=args.seed,
        pair_batch_size=args.pair_batch_size,
        window=args.window,
    )
    print(f"[probe] {len(ds.files)} walk files", flush=True)
    pair_ds = _PairsFromWalks(ds, window=args.window, rank=args.rank, world_size=args.world_size)
    if not args.no_prefetch:
        pair_ds = _ThreadPrefetch(pair_ds, depth=args.prefetch_depth)
    # pin_memory needs CUDA; the training run pins — that stage is not covered here.
    loader = DataLoader(pair_ds, batch_size=None, num_workers=0, pin_memory=False, persistent_workers=False)

    def report(tag: str, epoch: int, n: int, pairs: int, t0: float) -> None:
        rss, hwm = _host_mem_gib()
        dt = time.monotonic() - t0
        print(
            f"[probe] {tag} epoch {epoch} batch {n:,} pairs {pairs:,} | {n / dt if dt else 0:.1f} batches/s | "
            f"rss {rss:.2f} GiB peak {hwm:.2f} GiB | arrow allocated {pool.bytes_allocated() / 2**30:.2f} GiB "
            f"arrow max {pool.max_memory() / 2**30:.2f} GiB",
            flush=True,
        )

    for epoch in range(args.epochs):
        t0 = time.monotonic()
        n = pairs = 0
        report("start", epoch, n, pairs, t0)
        for center, context in loader:
            n += 1
            pairs += int(center.numel())
            if args.release_every and n % args.release_every == 0:
                pool.release_unused()
            if n % args.log_every == 0:
                report("tick", epoch, n, pairs, t0)
            if args.max_batches and n >= args.max_batches:
                break
        report("end", epoch, n, pairs, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
