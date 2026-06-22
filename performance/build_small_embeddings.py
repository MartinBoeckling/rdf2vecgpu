#!/usr/bin/env python
"""Filter full entity-embedding TSVs down to just the dataset (gold) entities,
producing KGrEAT's "small" embeddings so classification/regression tasks load a
few thousand rows instead of millions.

KGrEAT loads <type>_small.tsv from the embedding-small/ dir when a task requests
load_dataset_entities_only=True (classification & regression both do). The gold
entities are the non-empty `source` values in entity_mapping.tsv (full DBpedia
URIs), which match the full-URI keys in our embedding TSVs byte-for-byte.

Streaming (low RAM): keeps only lines whose first tab field is a gold URI.

Usage:  python build_small_embeddings.py <entity_mapping.tsv> <embedding_dir> <embedding_small_dir>
"""
import sys
import os
import glob
import pandas as pd


def main():
    mapping_tsv, emb_dir, small_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    em = pd.read_csv(mapping_tsv, sep="\t", dtype=str)
    src = {s for s in em["source"].dropna().tolist() if s != ""}
    print(f"gold entities (source set): {len(src)}")
    os.makedirs(small_dir, exist_ok=True)

    for f in sorted(glob.glob(os.path.join(emb_dir, "*.tsv"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(small_dir, f"{stem}_small.tsv")
        kept = total = 0
        with open(f) as fin, open(out, "w") as fout:
            for line in fin:
                total += 1
                key = line.split("\t", 1)[0]
                if key in src:
                    fout.write(line)
                    kept += 1
        print(f"  {stem}: kept {kept}/{total} rows -> {out}")


if __name__ == "__main__":
    main()
