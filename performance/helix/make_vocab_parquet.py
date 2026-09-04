#!/usr/bin/env python
"""Build a fit_from_walks-contract vocab.parquet from the resource repo's
partitioned word2idx.parquet directory.

Contract (see prepare_corpus / torchrec_word2vec): single parquet file,
columns `token_id: int64 not null`, `entity: string not null`, token_ids
contiguous 0..n-1.

Input parts have `word: string`, `token: int32`. ~393M rows for full
Wikidata — streamed part-by-part, constant memory. Run wherever the data
lives (dws before transfer, or Helix after):

  python performance/helix/make_vocab_parquet.py \
      /path/to/word2idx.parquet /path/to/corpus_dir/vocab.parquet
"""
import sys
import glob

import pyarrow as pa
import pyarrow.parquet as pq

src_dir, dst = sys.argv[1], sys.argv[2]
parts = sorted(glob.glob(f"{src_dir}/*.parquet"))
assert parts, f"no parquet parts under {src_dir}"

schema = pa.schema(
    [pa.field("token_id", pa.int64(), nullable=False),
     pa.field("entity", pa.string(), nullable=False)]
)
n = 0
with pq.ParquetWriter(dst, schema) as w:
    for p in parts:
        t = pq.read_table(p, columns=["word", "token"])
        out = pa.table(
            {"token_id": t.column("token").cast(pa.int64()),
             "entity": t.column("word")},
            schema=schema,
        )
        w.write_table(out)
        n += len(out)
print(f"wrote {dst}: {n:,} rows from {len(parts)} parts")
