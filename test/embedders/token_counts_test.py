"""CPU-only tests for the walk-corpus token counter + its on-disk cache
(embedders/word2vec_loader.py). Mirrors the real corpus schema: ``vertices``
list<int64>, ``predicates`` list<double>, ``-1`` sentinels on short walks."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rdf2vecgpu.embedders.word2vec_loader import (
    token_counts_from_walk_parquet,
    token_counts_cache_path,
    load_token_counts_cache,
    save_token_counts_cache,
)


def _write_walk_shards(walks_dir, rng, *, n_files=3, rows_per_file=400, depth=4, vocab=50):
    """Random fixed-length walks with -1 sentinel tails; returns the expected
    per-token counts computed the naive way (per-element Python loop)."""
    walks_dir.mkdir()
    expected = np.zeros(vocab, dtype=np.int64)
    for k in range(n_files):
        verts, preds = [], []
        for _ in range(rows_per_file):
            valid = int(rng.integers(1, depth + 2))            # 1..depth+1 real vertices
            v = list(rng.integers(0, vocab, size=valid)) + [-1] * (depth + 1 - valid)
            p = list(rng.integers(0, vocab, size=max(valid - 1, 0))) + [-1] * (depth - max(valid - 1, 0))
            verts.append([int(x) for x in v]); preds.append([float(x) for x in p])
            for x in v + p:
                if x >= 0:
                    expected[int(x)] += 1
        tbl = pa.table({
            "vertices": pa.array(verts, type=pa.list_(pa.int64())),
            "predicates": pa.array(preds, type=pa.list_(pa.float64())),
        })
        pq.write_table(tbl, walks_dir / f"part.{k}.parquet")
    return expected


def test_token_counts_match_naive_and_skip_sentinels(tmp_path):
    rng = np.random.default_rng(0)
    walks = tmp_path / "walks"
    expected = _write_walk_shards(walks, rng)
    got = token_counts_from_walk_parquet(str(walks), vocab_size=50)
    assert got.dtype == np.int64
    assert np.array_equal(got, expected)
    # ids >= vocab_size are dropped rather than crashing (stale/oversized vocab guard)
    got_small = token_counts_from_walk_parquet(str(walks), vocab_size=10)
    assert np.array_equal(got_small, expected[:10])


def test_token_counts_parallel_matches_serial(tmp_path):
    rng = np.random.default_rng(1)
    walks = tmp_path / "walks"
    expected = _write_walk_shards(walks, rng, n_files=5)
    got = token_counts_from_walk_parquet(str(walks), vocab_size=50, workers=2)
    assert np.array_equal(got, expected)


def test_token_counts_cache_roundtrip_and_stale_rejection(tmp_path):
    vocab = tmp_path / "vocab.parquet"
    pq.write_table(pa.table({"token_id": [0, 1, 2], "entity": ["a", "b", "c"]}), vocab)
    path = token_counts_cache_path(str(vocab))
    assert path == str(tmp_path / "token_counts.npy")

    assert load_token_counts_cache(path, 3) is None            # absent
    counts = np.array([5, 0, 7], dtype=np.int64)
    save_token_counts_cache(path, counts)
    assert not list(tmp_path.glob("*.tmp.npy"))                # atomic: no temp left behind
    assert np.array_equal(load_token_counts_cache(path, 3), counts)
    assert load_token_counts_cache(path, 4) is None            # wrong vocab size -> stale
    np.save(path, counts.astype(np.float32))
    assert load_token_counts_cache(path, 3) is None            # wrong dtype -> stale
