"""KGrEAT TSV straight from a torchrec DCP checkpoint — CPU only, no GPUs.

A sharded checkpoint (``<ckpt>/step_<N>/``) stores each rank's block of the
in-embedding table as ONE tensor at a known offset of ``__<rank>_0.distcp``
(``.metadata`` lists the row offset and size of every chunk). This script
reads one such block on the host, pairs it with its slice of
``vocab.parquet`` and writes the same rows the trainer's export would have
written — same filter, same L1 normalisation, same formatting (it calls
``bench_backends._write_kgreat_rows``) — into ``<out>.rank<r>``;
``--concat`` then joins the shards in rank order.

``--format parquet`` writes the *raw* vectors instead — float32 exactly as
trained, no normalisation, no rounding — as one Parquet file per rank,
``<out>/part-<r>.parquet`` with columns ``token_id`` (int64, the row of the
table), ``entity`` (IRI) and ``vector`` (``fixed_size_list<float32>[dim]``).
The directory is the dataset (``pyarrow.dataset`` / ``pandas.read_parquet``
read it whole); there is nothing to concatenate. This is the release format:
L1/L2 normalisation is lossy and the TSV's 6 decimals keep only 3-4
significant digits at d200.

Why: an export needs the whole 8-GPU allocation only because the model is
sharded there; the 2026-10-04 d200 run (14984051) finished training within
minutes of its 18 h limit and the alternative to this script was another
multi-day wait for 8x H200 just to write a file. One rank block of the
full-Wikidata table is 18.3 GiB (d100) / 36.6 GiB (d200); a cpu-single
node (236 GB) holds it twice over, and the eight ranks run as eight jobs.

  # one job per rank (needs ~2.2x the block size in RAM), then:
  python performance/helix/export_from_checkpoint.py --ckpt $OUT/ckpt --vocab $CORPUS/vocab.parquet \\
      --out $OUT/entities.tsv --entity-regex '^http://www.wikidata.org/entity/Q' --rank 3
  python performance/helix/export_from_checkpoint.py --ckpt $OUT/ckpt --out $OUT/entities.tsv --concat
  # raw float32 release files, one job per rank, no concat step:
  python performance/helix/export_from_checkpoint.py --ckpt $OUT/ckpt --vocab $CORPUS/vocab.parquet \\
      --out $OUT/entities_raw --format parquet --entity-regex '^http://www.wikidata.org/entity/Q' --rank 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # performance/


def _step_dir(ckpt: str) -> str:
    """``ckpt`` may be a step directory or a checkpoint root with ``latest``."""
    if os.path.exists(os.path.join(ckpt, ".metadata")):
        return ckpt
    latest = os.path.join(ckpt, "latest")
    if os.path.exists(latest):
        return os.path.realpath(latest)
    raise SystemExit(f"{ckpt}: neither a step directory (.metadata) nor a checkpoint root (latest)")


def _table_chunks(step: str, table: str):
    """``(fqn, [(row_start, rows, dim, file, offset, length), ...])`` for the
    ``in`` or ``out`` embedding table of the checkpoint, in row order."""
    from torch.distributed.checkpoint import FileSystemReader

    md = FileSystemReader(step).read_metadata()
    want = f"{table}_embeddings"
    fqns = [f for f, t in md.state_dict_metadata.items() if want in f and f.endswith("weight") and hasattr(t, "chunks")]
    if len(fqns) != 1:
        raise SystemExit(f"{step}: expected one '{want}' weight, found {fqns}")
    fqn = fqns[0]
    tm = md.state_dict_metadata[fqn]
    by_index = {idx.index: info for idx, info in md.storage_data.items() if idx.fqn == fqn}
    out = []
    for i, ch in enumerate(tm.chunks):
        info = by_index[i]
        out.append((int(ch.offsets[0]), int(ch.sizes[0]), int(ch.sizes[1]),
                    os.path.join(step, info.relative_path), int(info.offset), int(info.length)))
    out.sort()
    total = sum(c[1] for c in out)
    if total != int(tm.size[0]) or any(out[k][0] + out[k][1] != out[k + 1][0] for k in range(len(out) - 1)):
        raise SystemExit(f"{step}: chunks of {fqn} do not tile {tuple(tm.size)}")
    return fqn, out


def export_rank(args) -> int:
    import torch
    import torch.distributed.checkpoint.filesystem as dcp_fs

    from bench_backends import _load_entity_set, _write_kgreat_rows
    from rdf2vecgpu.embedders.torchrec_word2vec import _read_vocab_slice, _release_host_memory

    step = _step_dir(args.ckpt)
    fqn, chunks = _table_chunks(step, args.table)
    if not 0 <= args.rank < len(chunks):
        raise SystemExit(f"--rank {args.rank} out of range: checkpoint has {len(chunks)} chunks")
    start, rows, dim, path, offset, length = chunks[args.rank]
    if args.max_rows:
        rows_out = min(rows, args.max_rows)
    else:
        rows_out = rows
    print(f"[ckpt-export] {step} {fqn}: chunk {args.rank}/{len(chunks)} rows [{start:,}, {start + rows:,}) dim {dim} "
          f"<- {os.path.basename(path)} @ {offset} ({length / 2**30:.1f} GiB)", flush=True)

    t0 = time.monotonic()
    with open(path, "rb") as fh:
        view = dcp_fs._create_file_view(fh, offset, length)
        block = torch.load(view, map_location="cpu", weights_only=True)
    if tuple(block.shape) != (rows, dim):
        raise SystemExit(f"chunk {args.rank}: loaded {tuple(block.shape)}, metadata says {(rows, dim)}")
    print(f"[ckpt-export] block loaded in {time.monotonic() - t0:.0f} s, dtype {block.dtype}", flush=True)

    entities = _read_vocab_slice(args.vocab, start, start + rows_out)
    entity_set = _load_entity_set(args.entity_list)
    if args.format == "parquet":
        return _export_rank_parquet(args, block, entities, entity_set, start, rows_out, t0)
    shard = f"{args.out}.rank{args.rank}"
    os.makedirs(os.path.dirname(os.path.abspath(shard)) or ".", exist_ok=True)
    written = 0
    t1 = time.monotonic()
    with open(shard + ".tmp", "w") as out:
        for i in range(0, rows_out, args.chunk_rows):
            n = min(args.chunk_rows, rows_out - i)
            vectors = block[i:i + n].to(torch.float32).numpy()
            written += _write_kgreat_rows(entities.slice(i, n), vectors, out,
                                          entity_regex=args.entity_regex, entity_set=entity_set)
            if (i // args.chunk_rows) % 10 == 0:
                print(f"[ckpt-export] rank {args.rank}: {i + n:,}/{rows_out:,} block rows, {written:,} written, "
                      f"{time.monotonic() - t1:.0f} s", flush=True)
    os.replace(shard + ".tmp", shard)   # a shard only exists once it is complete
    del block
    _release_host_memory()
    print(f"[ckpt-export] rank {args.rank}: {written:,} of {rows_out:,} block rows -> {shard} "
          f"in {time.monotonic() - t0:.0f} s", flush=True)
    return 0


def _export_rank_parquet(args, block, entities, entity_set, start: int, rows_out: int, t0: float) -> int:
    """Raw float32 rows of one rank block -> ``<out>/part-<rank>.parquet``,
    one row group per ``--chunk-rows`` slice, same filters as the TSV path."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from bench_backends import _kgreat_keep_mask
    from rdf2vecgpu.embedders.torchrec_word2vec import _release_host_memory

    dim = int(block.shape[1])
    schema = pa.schema([("token_id", pa.int64()), ("entity", pa.large_string()),
                        ("vector", pa.list_(pa.float32(), dim))])
    os.makedirs(args.out, exist_ok=True)
    part = os.path.join(args.out, f"part-{args.rank:03d}.parquet")
    written = 0
    t1 = time.monotonic()
    with pq.ParquetWriter(part + ".tmp", schema, compression="zstd") as writer:
        for i in range(0, rows_out, args.chunk_rows):
            n = min(args.chunk_rows, rows_out - i)
            ents = entities.slice(i, n)
            vectors = block[i:i + n].to(torch.float32).numpy()
            ids = np.arange(start + i, start + i + n, dtype=np.int64)
            keep = _kgreat_keep_mask(ents, args.entity_regex, entity_set)
            if keep is not None:
                if not keep.any():
                    continue
                ents = ents.filter(pa.array(keep))
                vectors, ids = vectors[keep], ids[keep]
            vec_col = pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(vectors).reshape(-1)), dim)
            if isinstance(ents, pa.ChunkedArray):
                ents = ents.combine_chunks()
            writer.write_table(pa.table([pa.array(ids), ents.cast(pa.large_string()), vec_col], schema=schema))
            written += len(ids)
            if (i // args.chunk_rows) % 10 == 0:
                print(f"[ckpt-export] rank {args.rank}: {i + n:,}/{rows_out:,} block rows, {written:,} written, "
                      f"{time.monotonic() - t1:.0f} s", flush=True)
    os.replace(part + ".tmp", part)   # a part only exists once it is complete
    del block
    _release_host_memory()
    print(f"[ckpt-export] rank {args.rank}: {written:,} of {rows_out:,} block rows -> {part} "
          f"({os.path.getsize(part) / 1e9:.1f} GB) in {time.monotonic() - t0:.0f} s", flush=True)
    return 0


def concat(args) -> int:
    from bench_backends import _concat_rank_shards

    step = _step_dir(args.ckpt)
    _, chunks = _table_chunks(step, args.table)
    _concat_rank_shards(args.out, len(chunks))   # refuses if any rank shard is missing
    print(f"[ckpt-export] concatenated {len(chunks)} rank shards -> {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.1f} GB)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="step_<N> directory, or the checkpoint root (uses latest)")
    ap.add_argument("--out", required=True, help="final TSV path (rank shards are <out>.rank<r>); "
                                                 "with --format parquet, the output directory")
    ap.add_argument("--format", choices=("tsv", "parquet"), default="tsv",
                    help="tsv: KGrEAT TSV, L1-normalised, 6 decimals (what was scored); "
                         "parquet: raw float32 vectors, the release format")
    ap.add_argument("--vocab", help="corpus vocab.parquet (required with --rank)")
    ap.add_argument("--rank", type=int, help="export this rank's block to <out>.rank<r>")
    ap.add_argument("--concat", action="store_true", help="join all rank shards into <out>")
    ap.add_argument("--table", choices=("in", "out"), default="in")
    ap.add_argument("--entity-regex", default="")
    ap.add_argument("--entity-list", default="")
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--max-rows", type=int, default=0, help="only the first N block rows (validation runs)")
    args = ap.parse_args()
    if args.concat:
        if args.format == "parquet":
            ap.error("--format parquet writes a directory of parts; there is nothing to --concat")
        return concat(args)
    if args.rank is None or not args.vocab:
        ap.error("give --rank R --vocab vocab.parquet, or --concat")
    return export_rank(args)


if __name__ == "__main__":
    sys.exit(main())
