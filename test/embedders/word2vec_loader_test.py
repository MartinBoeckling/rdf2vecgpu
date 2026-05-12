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
