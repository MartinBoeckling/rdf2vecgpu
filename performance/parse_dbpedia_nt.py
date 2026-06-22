#!/usr/bin/env python
"""Parse DBpedia N-Triples/Turtle .ttl(.bz2) object-property dumps into a single
3-column (subject, predicate, object) parquet for the rdf2vecgpu walk pipeline.

CRITICAL — entity-key contract with KGrEAT:
The subject/object strings produced here MUST be byte-identical to the entity ids
that KGrEAT's identity mapping derives, otherwise downstream-task coverage drops
to zero. KGrEAT builds its ``known_entities`` set with
``shared/mapping/identity/importer.py::NTriplesEdgelistReader`` using the regex
``rb'<(.+)> <(.+)> <(.+)> \\.\\s*\\n'`` over the SAME files. We replicate that
exact object-triple pattern (URIs with angle brackets stripped, no namespace
shortening) so our embedding keys == KGrEAT's ``source`` == full URIs like
``http://dbpedia.org/resource/Andrei_Tarkovsky``.

Usage:
  python parse_dbpedia_nt.py OUT.parquet IN1.ttl.bz2 [IN2.ttl.bz2 ...]
"""
import sys
import bz2
import re
import pyarrow as pa
import pyarrow.parquet as pq

# Identical to KGrEAT's NTriplesEdgelistReader object_pattern.
OBJECT_PATTERN = re.compile(rb'<(.+)> <(.+)> <(.+)> \.\s*\n')

BATCH = 2_000_000  # triples per parquet row-group


def _open(path):
    return bz2.open(path, "rb") if path.endswith(".bz2") else open(path, "rb")


def main():
    out_path = sys.argv[1]
    in_paths = sys.argv[2:]
    schema = pa.schema([("subject", pa.string()),
                        ("predicate", pa.string()),
                        ("object", pa.string())])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")

    subs, preds, objs = [], [], []
    n_triples = 0
    n_lines = 0
    n_skipped = 0

    def flush():
        nonlocal subs, preds, objs
        if not subs:
            return
        tbl = pa.table({"subject": subs, "predicate": preds, "object": objs}, schema=schema)
        writer.write_table(tbl)
        subs, preds, objs = [], [], []

    for path in in_paths:
        print(f"[parse] reading {path}", flush=True)
        with _open(path) as fh:
            for line in fh:
                n_lines += 1
                m = OBJECT_PATTERN.match(line)
                if m is None:
                    n_skipped += 1
                    continue
                s, p, o = (x.decode("utf-8") for x in m.groups())
                subs.append(s); preds.append(p); objs.append(o)
                n_triples += 1
                if len(subs) >= BATCH:
                    flush()
                if n_lines % 5_000_000 == 0:
                    print(f"[parse]   lines={n_lines:,} triples={n_triples:,} "
                          f"skipped={n_skipped:,}", flush=True)
        flush()

    flush()
    writer.close()
    print(f"[parse] DONE -> {out_path}", flush=True)
    print(f"[parse]   total lines={n_lines:,} object-triples={n_triples:,} "
          f"non-object-skipped={n_skipped:,}", flush=True)


if __name__ == "__main__":
    main()
