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


def test_walkseq_iterable_rank_striping_disjoint_equal_counts(walk_seq_dir):
    """Multi-GPU DDP: each rank must train on a DISJOINT 1/world_size row-shard,
    and every rank must emit the SAME number of batches. pytorch-mg uses
    Lightning's automatic loop (no per-step all_reduce join available), so equal
    batch counts must hold by construction or the per-step gradient all_reduce
    desyncs and the job hangs. 1000 walks / 4 ranks → 250 rows each, divisible."""
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    ws = 4
    per_rank_ids, batch_counts = {}, {}
    for rank in range(ws):
        it = _WalkSeqIterable(walk_seq_dir, emit_batch_size=64, read_chunk_size=256,
                              shuffle=False, rank=rank, world_size=ws)
        batches = list(iter(it))
        batch_counts[rank] = len(batches)
        per_rank_ids[rank] = (
            np.concatenate([b[0][:, 0] // 10 for b in batches])
            if batches else np.array([], dtype=np.int32)
        )
    # equal batch counts across ranks — the DDP requirement
    assert len(set(batch_counts.values())) == 1, batch_counts
    # each rank gets exactly 1000/4 = 250 distinct walk rows
    for rank, ids in per_rank_ids.items():
        assert len(ids) == 250 and len(np.unique(ids)) == 250
    # ranks are disjoint and together cover all 1000 walks
    allids = np.concatenate(list(per_rank_ids.values()))
    assert len(allids) == 1000 and len(np.unique(allids)) == 1000


def test_walkseq_iterable_rank_striping_equal_counts_when_indivisible(tmp_path):
    """When rows don't divide evenly across ranks, every rank STILL emits the
    same batch count (the < world_size remainder rows are dropped). 10 walks /
    3 ranks → 3 rows each (9 used, 1 dropped), 2 batches each at emit=2."""
    import pyarrow as pa
    d = tmp_path / "walks"; d.mkdir()
    n = 10
    vs = np.arange(n, dtype=np.int32)[:, None] * 10 + np.arange(2, dtype=np.int32)[None, :]
    ps = np.arange(n, dtype=np.int32)[:, None] * 10 + 100000
    pq.write_table(pa.table({
        "vertices": pa.array(list(vs), type=pa.list_(pa.int32())),
        "predicates": pa.array(list(ps), type=pa.list_(pa.int32())),
    }), d / "part.0.parquet")

    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    ws = 3
    per_rank_ids, batch_counts = {}, {}
    for rank in range(ws):
        batches = list(iter(_WalkSeqIterable(str(d), emit_batch_size=2,
                          read_chunk_size=4, shuffle=False, rank=rank, world_size=ws)))
        batch_counts[rank] = len(batches)
        per_rank_ids[rank] = np.concatenate([b[0][:, 0] // 10 for b in batches])
    assert len(set(batch_counts.values())) == 1, batch_counts        # equal counts
    for ids in per_rank_ids.values():
        assert len(ids) == 3 and len(np.unique(ids)) == 3            # 10//3 each
    allids = np.concatenate(list(per_rank_ids.values()))
    assert len(allids) == 9 and len(np.unique(allids)) == 9          # disjoint, 1 dropped


def test_walkseq_iterable_shuffle_deterministic_per_seed(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    a = np.concatenate([b[0] for b in _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=3)], axis=0)
    b = np.concatenate([bb[0] for bb in _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=3)], axis=0)
    c = np.concatenate([cc[0] for cc in _WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=True, seed=99)], axis=0)
    np.testing.assert_array_equal(a, b)          # same seed → same order
    assert not np.array_equal(a, c)              # different seed → different order


import pytest


@pytest.mark.gpu
def test_inmemory_walkseq_yields_gpu_tensors_of_right_shape():
    import cudf
    import torch
    from rdf2vecgpu.embedders.word2vec_loader import InMemoryWalkSeqDataModule

    # 5 walks, depth=3 → vertices length 4, predicates length 3 (cuGraph -1 pads).
    df = cudf.DataFrame({
        "vertices":   [[0, 1, 2, 3], [1, 2, 3, -1], [2, 0, -1, -1],
                       [3, 1, 0, 2], [0, 0, 1, -1]],
        "predicates": [[10, 11, 12], [11, 12, -1], [12, -1, -1],
                       [10, 10, 11], [11, 12, -1]],
    })
    dm = InMemoryWalkSeqDataModule(df, emit_batch_size=2, shuffle=False)
    dm.setup()
    batches = list(dm.train_dataloader())

    # 5 rows, emit_batch_size=2 → 3 batches (2, 2, 1).
    assert [b[0].shape[0] for b in batches] == [2, 2, 1]
    v0, p0 = batches[0]
    assert v0.is_cuda and p0.is_cuda            # GPU-resident, no CPU round-trip
    assert v0.shape == (2, 4) and p0.shape == (2, 3)
    # Round-trips the data unchanged (shuffle=False).
    assert v0[0].tolist() == [0, 1, 2, 3]
    assert p0[0].tolist() == [10, 11, 12]


# ───────── pair-budget batching + prefetch (torchrec loader, 2026-09-06) ─────────


def test_pairs_by_valid_len_matches_full_walk_formula():
    from rdf2vecgpu.embedders.word2vec_loader import _pairs_by_valid_len, skipgram_pairs_per_walk
    t = _pairs_by_valid_len(9, 5)                       # depth 4 → 9 tokens, window 5
    assert t[9] == skipgram_pairs_per_walk(4, 5) == 60
    assert list(t[:4]) == [0, 0, 2, 6]                  # stub, 1 vertex, v-p, v-p-v
    assert list(_pairs_by_valid_len(5, 2)) == [0, 0, 2, 6, 10, 14]
    assert _pairs_by_valid_len(5, 2)[5] == skipgram_pairs_per_walk(2, 2)


def _walks_with_sentinels(n=600, depth_p1=5, seed=0):
    """cuGraph-style walks: row r keeps its first `first[r]` vertices (1..depth+1)
    and `first[r]-1` predicates; the rest are -1 sentinels."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 50, size=(n, depth_p1)).astype(np.int32)
    p = rng.integers(100, 150, size=(n, depth_p1 - 1)).astype(np.int32)
    first = rng.integers(1, depth_p1 + 1, size=n)
    for r in range(n):
        v[r, first[r]:] = -1
        p[r, first[r] - 1:] = -1
    return v, p


def test_walk_pair_counts_match_pair_expansion():
    import torch
    from rdf2vecgpu.embedders.word2vec_loader import _PairsFromWalks, _walk_pair_counts
    v, p = _walks_with_sentinels()
    counts = _walk_pair_counts(v, window=5)
    assert counts.min() == 0 and counts.max() == 60            # stubs and full walks both present
    (c, x), = list(_PairsFromWalks([(v, p)], window=5))
    assert c.dtype == torch.int64 and c.numel() == x.numel() == int(counts.sum())
    assert not ((c == -1) | (x == -1)).any()                    # sentinels never leak into pairs
    for r in range(0, len(v), 41):                              # per-row agreement
        (cr, _), = list(_PairsFromWalks([(v[r:r + 1], p[r:r + 1])], window=5))
        assert cr.numel() == counts[r], r
    # rank striding: the union over ranks is exactly the single-rank output
    parts = [list(_PairsFromWalks([(v, p)], window=5, rank=k, world_size=3))[0] for k in range(3)]
    assert sum(cp.numel() for cp, _ in parts) == c.numel()


def test_pair_budget_batching_preserves_rows_and_hits_budget(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable, _walk_pair_counts
    rows_mode = list(_WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=False))
    pair_mode = list(_WalkSeqIterable(walk_seq_dir, emit_batch_size=128, read_chunk_size=256, shuffle=False,
                                      pair_batch_size=1000, window=5))
    # same rows, same order (batches may span read chunks)
    for col in (0, 1):
        np.testing.assert_array_equal(np.concatenate([b[col] for b in pair_mode]),
                                      np.concatenate([b[col] for b in rows_mode]))
    per_walk = int(_walk_pair_counts(rows_mode[0][0][:1], 5)[0])       # fixture walks are full: 40
    assert per_walk == 40
    counts = [int(_walk_pair_counts(b[0], 5).sum()) for b in pair_mode]
    assert len(counts) == 40 and all(1000 <= c < 1000 + per_walk for c in counts[:-1])
    assert 0 < counts[-1] <= 1000 + per_walk
    # rows-mode batch count (8) != pair-mode batch count (40): the budget rules


def test_pair_budget_batcher_handles_stubs_and_chunk_boundaries():
    from rdf2vecgpu.embedders.word2vec_loader import _PairBudgetBatcher, _walk_pair_counts
    v, p = _walks_with_sentinels(n=1000, seed=3)
    b = _PairBudgetBatcher(pair_batch_size=333, window=5)
    out = []
    for s in range(0, 1000, 70):                                   # ragged pushes
        out += list(b.push(v[s:s + 70], p[s:s + 70]))
    out += list(b.flush())
    np.testing.assert_array_equal(np.concatenate([o[0] for o in out]), v)
    np.testing.assert_array_equal(np.concatenate([o[1] for o in out]), p)
    counts = [int(_walk_pair_counts(o[0], 5).sum()) for o in out]
    assert all(333 <= c < 333 + 60 for c in counts[:-1]) and 0 <= counts[-1] < 333 + 60


def test_pair_budget_rejects_in_iterable_ddp_striding(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    with pytest.raises(ValueError, match="world_size=1"):
        _WalkSeqIterable(walk_seq_dir, emit_batch_size=8, pair_batch_size=100, world_size=2, rank=0)


def test_thread_prefetch_preserves_order_and_propagates_errors():
    from rdf2vecgpu.embedders.word2vec_loader import _ThreadPrefetch
    items = [(np.full((2, 2), i, dtype=np.int32), np.full((2, 1), i, dtype=np.int32)) for i in range(50)]
    out = list(_ThreadPrefetch(items, depth=3))
    assert len(out) == 50 and [int(o[0][0, 0]) for o in out] == list(range(50))

    class _Bad:
        def __iter__(self):
            yield items[0]
            raise RuntimeError("boom")

    it = iter(_ThreadPrefetch(_Bad(), depth=2))
    next(it)
    with pytest.raises(RuntimeError, match="boom"):
        next(it)
    it2 = iter(_ThreadPrefetch(items, depth=2))                  # early stop must not hang
    next(it2)
    it2.close()


def _write_walks(d, v, p):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d.mkdir(exist_ok=True)
    pq.write_table(pa.table({
        "vertices": pa.array(list(v), type=pa.list_(pa.int32())),
        "predicates": pa.array(list(p), type=pa.list_(pa.int32())),
    }), d / "part.0.parquet")
    return str(d)


def test_pair_mode_drops_stubs_without_changing_the_pair_stream(tmp_path):
    """Stubs (walks with < 2 valid vertices) yield no pairs. Dropping them at
    decode must remove exactly those rows, keep the order of the rest, and
    leave the batch sequence (pairs per batch, batch count) identical."""
    from rdf2vecgpu.embedders.word2vec_loader import (
        _PairsFromWalks, _WalkSeqIterable, _walk_pair_counts, _walk_valid_len,
    )
    v, p = _walks_with_sentinels(n=800, seed=5)          # ~20% stubs (1 valid vertex)
    d = _write_walks(tmp_path / "walks", v, p)
    stubs = _walk_valid_len(v) < 3
    assert 100 < stubs.sum() < 250

    kw = dict(emit_batch_size=64, read_chunk_size=256, shuffle=False, pair_batch_size=500, window=5)
    keep_it = _WalkSeqIterable(d, drop_stubs=False, **kw)
    drop_it = _WalkSeqIterable(d, **kw)                   # default: on in pair-budget mode
    assert drop_it.drop_stubs and not keep_it.drop_stubs
    keep_b, drop_b = list(keep_it), list(drop_it)

    np.testing.assert_array_equal(np.concatenate([b[0] for b in keep_b]), v)
    np.testing.assert_array_equal(np.concatenate([b[0] for b in drop_b]), v[~stubs])
    np.testing.assert_array_equal(np.concatenate([b[1] for b in drop_b]), p[~stubs])
    assert all((_walk_valid_len(b[0]) >= 3).all() for b in drop_b)
    # same batch sequence: stubs contribute 0 pairs, so the cut points coincide
    assert [int(_walk_pair_counts(b[0], 5).sum()) for b in keep_b] == \
           [int(_walk_pair_counts(b[0], 5).sum()) for b in drop_b]
    # and the expanded pairs are identical batch by batch
    for (ck, xk), (cd, xd) in zip(_PairsFromWalks(keep_b, window=5), _PairsFromWalks(drop_b, window=5)):
        np.testing.assert_array_equal(ck.numpy(), cd.numpy())
        np.testing.assert_array_equal(xk.numpy(), xd.numpy())


def test_drop_stubs_is_off_in_row_count_mode_and_rejects_ddp(walk_seq_dir):
    from rdf2vecgpu.embedders.word2vec_loader import _WalkSeqIterable
    assert not _WalkSeqIterable(walk_seq_dir, emit_batch_size=128).drop_stubs
    with pytest.raises(ValueError, match="world_size=1"):
        _WalkSeqIterable(walk_seq_dir, emit_batch_size=8, drop_stubs=True, world_size=2, rank=1)
