"""Tests for weighted walks and walk boundary conditions.

cuGraph's biased walks return the traversed edge's weight per step, not its
predicate; `recover_weighted_predicates` maps (src, dst, weight) back to a
predicate, drawing uniformly among parallel edges of equal weight. The graph
used here has, from vertex 0: an exact duplicate triple, two parallel edges of
equal weight (a tie), parallel edges of different weight and a second
neighbour; a self-loop at 3; and dead ends at 1 and 2. All predicate tokens
are above 2^24, so a float32 round trip would change them.
"""
from __future__ import annotations

import math

import cupy as cp
import cudf
import numpy as np
import pytest

from src.rdf2vecgpu.corpus.walk_corpus import (
    SingleGPUWalkCorpus,
    _decode_predicates,
    _walk_stepwise,
    encode_predicates,
    predicate_code_table,
    recover_weighted_predicates,
    weighted_edge_index,
)

BIG = 300_000_001
A, B, C, D, SELF, EXIT = (BIG + i for i in range(6))
# weighted: out-weight of 0 is 1 + 1 + 3 + 3 + 4 = 12
WEIGHTED_FROM_0 = {A: 2 / 12, B: 3 / 12, D: 3 / 12, C: 4 / 12}
UNIFORM_FROM_0 = {A: 2 / 5, B: 1 / 5, D: 1 / 5, C: 1 / 5}


def _edges(wdtype="float32"):
    return cudf.DataFrame({
        "subject": cudf.Series([0, 0, 0, 0, 0, 3, 3], dtype="int32"),
        "predicate": cudf.Series([A, A, B, D, C, SELF, EXIT], dtype="int64"),
        "object": cudf.Series([1, 1, 1, 1, 2, 3, 0], dtype="int32"),
        "weights": cudf.Series([1, 1, 3, 3, 4, 1, 1], dtype=wdtype),
    })


def _corpus(edges, weighted):
    from cugraph import MultiGraph

    G = MultiGraph(directed=True)
    if weighted:
        G.from_cudf_edgelist(edges, source="subject", destination="object", weight="weights", renumber=False)
        return SingleGPUWalkCorpus(G, window_size=2, walk_weighted=True, edge_index=weighted_edge_index(edges))
    tokens = edges["predicate"].unique().sort_values().values_host
    edges = edges.assign(pcode=encode_predicates(edges["predicate"], tokens))
    G.from_cudf_edgelist(edges, source="subject", destination="object", edge_attr="pcode", renumber=False)
    return SingleGPUWalkCorpus(G, window_size=2, code_to_token=predicate_code_table(tokens))


def _matrices(walks, depth):
    V = walks["vertices"].list.leaves.values_host.reshape(len(walks), depth + 1)
    P = walks["predicates"].list.leaves.values_host.reshape(len(walks), depth)
    return V, P


def _assert_frequencies(values, expected, n):
    got = dict(zip(*np.unique(values, return_counts=True)))
    assert set(got) == set(expected), got
    for t, p in expected.items():
        assert abs(got[t] / n - p) < 5 * math.sqrt(p * (1 - p) / n), (t, got[t] / n, p)


def _assert_valid_steps(V, P, edges):
    legit = {(int(r.subject), int(r.predicate), int(r.object)) for r in edges.to_pandas().itertuples()}
    m = V[:, 1:] >= 0
    steps = set(zip(V[:, :-1][m].tolist(), P[m].tolist(), V[:, 1:][m].tolist()))
    assert steps and steps <= legit, steps - legit
    assert (P[~m] == 0).all(), "predicate padding after a walk's end"
    assert (V[:, 1:][V[:, :-1] < 0] < 0).all(), "a walk resumes after its end"
    return steps


@pytest.mark.gpu
def test_edge_index_groups_parallel_edges():
    idx = weighted_edge_index(_edges()).to_pandas()
    groups = idx.groupby(["subject", "object", "weights"])
    assert sorted(groups.size().tolist()) == [1, 1, 1, 2, 2]
    for _, g in groups:
        assert sorted(g["rank"]) == list(range(len(g))) and (g["n"] == len(g)).all()
    tie = idx[(idx.subject == 0) & (idx.object == 1) & (idx.weights == 3)]
    assert set(tie.predicate) == {B, D}


@pytest.mark.gpu
@pytest.mark.parametrize("wdtype", ["float32", "float64"])
def test_recover_draws_among_equal_weight_parallel_edges(wdtype):
    edges = _edges(wdtype)
    n = 40_000
    steps = cudf.DataFrame({
        "walk_id": cp.arange(n + 2, dtype="int64"),
        "src": cp.zeros(n + 2, dtype="int32"),
        "weight": cp.concatenate([cp.full(n, 3), cp.array([1, 4])]).astype(wdtype),
        "dst": cp.concatenate([cp.ones(n), cp.array([1, 2])]).astype("int32"),
    })
    out = recover_weighted_predicates(steps, weighted_edge_index(edges), random_state=0)
    assert list(out.columns) == ["walk_id", "src", "predicate", "dst"]
    assert str(out["predicate"].dtype) == "int64"
    out = out.sort_values("walk_id").to_pandas()
    assert out.predicate.iloc[-2:].tolist() == [A, C]
    _assert_frequencies(out.predicate.iloc[:-2].to_numpy(), {B: 0.5, D: 0.5}, n)


@pytest.mark.gpu
def test_recover_rejects_steps_without_edge():
    steps = cudf.DataFrame({"src": [0], "dst": [1], "weight": cudf.Series([2.0], dtype="float32")})
    with pytest.raises(ValueError, match="match no"):
        recover_weighted_predicates(steps, weighted_edge_index(_edges()))


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("wdtype", ["float32", "float64"])
def test_walk_sequences_follow_edges(weighted, wdtype):
    """First steps from 0 follow the edge weights (or are uniform over the
    parallel edges); every step is an input triple; dead ends pad."""
    if not weighted and wdtype == "float64":
        pytest.skip("weights unused")
    edges = _edges(wdtype)
    n = 40_000
    walks = _corpus(edges, weighted).random_walk_sequences(
        cudf.Series(np.zeros(n), dtype="int32"), walk_depth=2, random_state=0
    )
    V, P = _matrices(walks, 2)
    _assert_valid_steps(V, P, edges)
    _assert_frequencies(P[:, 0], WEIGHTED_FROM_0 if weighted else UNIFORM_FROM_0, n)
    assert (V[:, 2] == -1).all() and (P[:, 1] == 0).all()  # 1 and 2 are dead ends


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [True, False])
def test_skipgram_pairs_follow_edges(weighted):
    edges = _edges()
    n = 20_000
    pairs = _corpus(edges, weighted).random_walk(
        edge_df=edges, walk_vertices=cudf.Series(np.zeros(n), dtype="int32"), walk_depth=2,
        random_state=0, word2vec_model="skipgram", min_count=1,
    ).to_pandas()
    # walks are [0, p, 1|2]: from center 0, one predicate context per walk
    from_0 = pairs[(pairs.center == 0) & (pairs.context >= BIG)].context.to_numpy()
    assert len(from_0) == n
    _assert_frequencies(from_0, WEIGHTED_FROM_0 if weighted else UNIFORM_FROM_0, n)


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("depth", [1, 5])
def test_dead_ends_and_self_loops(weighted, depth):
    edges = _edges()
    starts = cudf.Series([1, 2] * 10 + [3] * 2_000, dtype="int32")
    walks = _corpus(edges, weighted).random_walk_sequences(starts, walk_depth=depth, random_state=1)
    V, P = _matrices(walks, depth)
    assert V.shape == (len(starts), depth + 1)
    steps = _assert_valid_steps(V, P, edges)
    sinks = np.isin(V[:, 0], [1, 2])
    assert (V[sinks, 1:] == -1).all() and (P[sinks] == 0).all()
    assert (3, SELF, 3) in steps
    # a walk stops early only at a dead end
    last = V[np.arange(len(V)), (V >= 0).sum(1) - 1]
    assert np.isin(last[(V < 0).any(1)], [1, 2]).all()


@pytest.mark.gpu
def test_predicate_codes_exact_at_float32_limit():
    """2^24 - 1 predicates are the most that get float32-exact codes; the
    highest code still decodes to its token, one more predicate is refused."""
    tokens = np.arange(2**24 - 1, dtype=np.int64) * 3 + 7
    table = predicate_code_table(tokens)
    probe = cudf.Series(np.concatenate([tokens[:3], tokens[-3:]]))
    codes = encode_predicates(probe, tokens)
    assert codes.dtype == np.float32 and int(codes.max()) == 2**24 - 1
    assert (_decode_predicates(codes.values, table).get() == probe.values_host).all()
    with pytest.raises(ValueError, match="float32-exact"):
        predicate_code_table(np.arange(2**24, dtype=np.int64))


@pytest.mark.gpu
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_weights_rejected(bad):
    from src.rdf2vecgpu.gpu_rdf2vec import _checked_weights

    edges = _edges()
    edges["weights"] = cudf.Series([1, 1, 3, 3, 4, 1, bad], dtype="float32", nan_as_null=False)
    with pytest.raises(ValueError, match="1 edge weights"):
        _checked_weights(edges)


@pytest.mark.gpu
def test_integer_weights_become_float():
    from src.rdf2vecgpu.gpu_rdf2vec import _checked_weights

    edges = _edges().astype({"weights": "int32"})
    assert str(_checked_weights(edges)["weights"].dtype) == "float32"


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [True, False])
def test_walks_when_every_walk_ends_early(weighted):
    """cuGraph's single-GPU biased walks fail once every walk has ended two
    or more steps before max_depth; the corpus then walks step by step."""
    edges = _edges()
    n, depth = 2_000, 4
    corpus = _corpus(edges, weighted)
    walks = corpus.random_walk_sequences(cudf.Series(np.zeros(n), dtype="int32"), walk_depth=depth, random_state=0)
    V, P = _matrices(walks, depth)
    _assert_valid_steps(V, P, edges)
    assert (V[:, 2:] == -1).all() and (P[:, 1:] == 0).all()
    _assert_frequencies(P[:, 0], WEIGHTED_FROM_0 if weighted else UNIFORM_FROM_0, n)
    sinks = corpus.random_walk_sequences(cudf.Series([1, 2] * 10, dtype="int32"), walk_depth=depth, random_state=0)
    V, P = _matrices(sinks, depth)
    assert (V[:, 1:] == -1).all() and (P == 0).all()
    pairs = corpus.random_walk(edge_df=edges, walk_vertices=cudf.Series(np.zeros(n), dtype="int32"),
                               walk_depth=depth, random_state=0, word2vec_model="skipgram", min_count=1)
    assert len(pairs[(pairs.center == 0) & (pairs.context >= BIG)]) == n


@pytest.mark.gpu
def test_walk_stepwise_matches_walk_probabilities():
    """From 3 (self-loop and exit, equal weight) a walk is still at 3 after k
    steps with probability 2^-k; from 0 the first step follows the weights."""
    from cugraph import MultiGraph, biased_random_walks

    edges = _edges()
    G = MultiGraph(directed=True)
    G.from_cudf_edgelist(edges, source="subject", destination="object", weight="weights", renumber=False)

    def one_step(starts, seed):
        v, w, _ = biased_random_walks(G, start_vertices=cudf.Series(starts), max_depth=1, random_state=seed)
        return v.values, w.values

    n, depth = 40_000, 3
    starts = cp.concatenate([cp.full(n, 3), cp.zeros(n)]).astype(cp.int32)
    v, w = _walk_stepwise(one_step, starts, depth, random_state=0)
    V = v.reshape(-1, depth + 1).get()
    W = w.reshape(-1, depth).get()
    assert (V[:, 0] == starts.get()).all()
    loop = V[:n]
    for k in (1, 2, 3):
        p = 0.5 ** k
        f = (loop[:, 1:k + 1] == 3).all(1).mean()
        assert abs(f - p) < 5 * math.sqrt(p * (1 - p) / n), (k, f)
    first = V[n:, 1]
    weights = W[n:, 0]
    # 0 -> 1 carries weight 1 + 1 + 3 + 3 = 8 of 12, 0 -> 2 weight 4
    assert abs((first == 1).mean() - 8 / 12) < 5 * math.sqrt(8 / 12 * 4 / 12 / n)
    assert set(np.unique(weights[first == 2]).tolist()) == {4.0}
    assert (V[n:, 2:] == -1).all() and (W[n:, 1:] == 0).all()
