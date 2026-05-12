from rdf2vecgpu.embedders.word2vec_loader import (
    skipgram_pairs_per_walk,
    walks_per_pair_batch,
)


def test_skipgram_pairs_per_walk_known_values():
    # interleaved length L = 2*depth+1; a pair (i, j) exists for every
    # 0 <= i, j < L with 0 < |i - j| <= window.
    # depth 4, window 5  → L = 9  → 60 ordered pairs (window covers everything
    #   except the i=0..j=6 / i=8..j=2 corners).
    assert skipgram_pairs_per_walk(4, 5) == 60
    # depth 2, window 2  → L = 5  → 14 ordered pairs.
    assert skipgram_pairs_per_walk(2, 2) == 14
    # depth 1, window 5  → L = 3  → every pair: 3*2 = 6.
    assert skipgram_pairs_per_walk(1, 5) == 6


def test_walks_per_pair_batch():
    # 16384 pairs / 60 pairs-per-walk → 273 walks/batch (floor; short walks
    # produce fewer pairs so the realised batch is <= the 16384 target).
    assert walks_per_pair_batch(16384, 4, 5) == 273
    # never returns 0 even if pair_batch_size < pairs_per_walk.
    assert walks_per_pair_batch(10, 4, 5) == 1


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def walk_seq_dir(tmp_path):
    """One parquet 'part' file: 1000 walks, walk_depth=3 ⇒ vertices len 4,
    predicates len 3. Walk indices encode their row number so we can check
    re-slicing preserves order."""
    d = tmp_path / "walks"
    d.mkdir()
    n = 1000
    vs = (np.arange(n, dtype=np.int32)[:, None] * 10 + np.arange(4, dtype=np.int32)[None, :])
    ps = (np.arange(n, dtype=np.int32)[:, None] * 10 + np.arange(3, dtype=np.int32)[None, :] + 100000)
    tbl = pa.table({
        "vertices": pa.array(list(vs), type=pa.list_(pa.int32())),
        "predicates": pa.array(list(ps), type=pa.list_(pa.int32())),
    })
    pq.write_table(tbl, d / "part.0.parquet")
    return str(d)


def test_walkseq_iterable_reslices_in_order(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    it = _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=False)
    batches = list(iter(it))
    # 1000 rows / 128 → 8 batches (sizes 128*7 + 104), regardless of the 256 read chunk.
    assert [b[0].shape[0] for b in batches] == [128, 128, 128, 128, 128, 128, 128, 104]
    v_all = np.concatenate([b[0] for b in batches], axis=0)
    p_all = np.concatenate([b[1] for b in batches], axis=0)
    assert v_all.dtype == np.int32 and p_all.dtype == np.int32
    assert v_all.shape == (1000, 4) and p_all.shape == (1000, 3)
    # rows came back in file order
    np.testing.assert_array_equal(v_all[:, 0], np.arange(1000, dtype=np.int32) * 10)
    np.testing.assert_array_equal(p_all[:, 0], np.arange(1000, dtype=np.int32) * 10 + 100000)


def test_walk_sequence_datamodule_small_batches(walk_seq_dir):
    import torch
    from rdf2vecgpu.embedders.word2vec_loader import WalkSequenceDataModule
    dm = WalkSequenceDataModule(walk_seq_dir, batch_size=128, num_workers=0, read_chunk_size=256, shuffle=False)
    batches = list(dm.train_dataloader())
    assert len(batches) == 8
    v, p = batches[0]
    assert isinstance(v, torch.Tensor) and v.dtype == torch.int32
    assert v.shape == (128, 4) and p.shape == (128, 3)
    v_last, _ = batches[-1]
    assert v_last.shape[0] == 104


def test_walkseq_iterable_shuffle_permutes_within_chunk_but_keeps_rows(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    # shuffle=True (default): rows come back permuted, but the *set* of rows and
    # the per-row (vertices, predicates) pairing are preserved.
    it = _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=7)
    batches = list(iter(it))
    v_all = np.concatenate([b[0] for b in batches], axis=0)
    p_all = np.concatenate([b[1] for b in batches], axis=0)
    assert v_all.shape == (1000, 4) and p_all.shape == (1000, 3)
    # not in file order anymore
    assert not np.array_equal(v_all[:, 0], np.arange(1000, dtype=np.int32) * 10)
    # same multiset of walk ids, and vertices[:,0]//10 == predicates[:,0]//10 - 10000//... pairing intact:
    walk_ids_v = np.sort(v_all[:, 0] // 10)
    np.testing.assert_array_equal(walk_ids_v, np.arange(1000, dtype=np.int32))
    # for each row, vertices[0]//10 (the walk id) must equal (predicates[0]-100000)//10 (same walk id)
    np.testing.assert_array_equal(v_all[:, 0] // 10, (p_all[:, 0] - 100000) // 10)


def test_walkseq_iterable_shuffle_deterministic_per_seed(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    a = np.concatenate([b[0] for b in _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=3)], axis=0)
    b = np.concatenate([bb[0] for bb in _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=3)], axis=0)
    c = np.concatenate([cc[0] for cc in _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=99)], axis=0)
    np.testing.assert_array_equal(a, b)          # same seed → same order
    assert not np.array_equal(a, c)              # different seed → different order
