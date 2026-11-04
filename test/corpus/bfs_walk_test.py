"""Tests for BFS walks: root-to-leaf paths of the BFS tree, in path order,
with one predicate per step.

Previously each walk's edges were ordered by subject token instead of depth
(scrambling paths whose vertex ids do not increase), and the predicate was
merged back by (subject, object), duplicating a step once per parallel edge.
Graph: 5 -> 3 over two parallel predicates, 3 -> 1, 5 -> 4, a back edge
1 -> 5 and a self-loop at 4; vertex ids decrease along 5 -> 3 -> 1.
"""
from __future__ import annotations

import math

import cudf
import numpy as np
import pytest

from src.rdf2vecgpu.corpus.walk_corpus import SingleGPUWalkCorpus, _bfs_walk_steps


def _edges():
    return cudf.DataFrame({
        "subject": cudf.Series([5, 5, 3, 5, 1, 4], dtype="int32"),
        "predicate": cudf.Series([100, 103, 101, 102, 104, 105], dtype="int32"),
        "object": cudf.Series([3, 3, 1, 4, 5, 4], dtype="int32"),
    })


def _graph(edges):
    from cugraph import MultiGraph

    G = MultiGraph(directed=True)
    G.from_cudf_edgelist(edges, source="subject", destination="object", renumber=False)
    return G


def _walks(steps):
    """walk_id → (vertex path, predicate path), checking step continuity."""
    walks = {}
    for wid, g in steps.to_pandas().groupby("walk_id"):
        g = g.sort_values("step")
        assert g["step"].tolist() == list(range(len(g))), "one row per step"
        assert (g["dst"].to_numpy()[:-1] == g["src"].to_numpy()[1:]).all(), "steps chain"
        walks[wid] = (tuple([int(g["src"].iloc[0])] + g["dst"].astype(int).tolist()), tuple(g["predicate"].astype(int)))
    return walks


@pytest.mark.gpu
def test_bfs_paths_in_order_from_the_root():
    edges = _edges()
    walks = _walks(_bfs_walk_steps(edges, cudf.Series([5], dtype="int32"), walk_depth=3, random_state=0))
    assert sorted(v for v, _ in walks.values()) == [(5, 3, 1), (5, 4)]
    preds = {v: p for v, p in walks.values()}
    assert preds[(5, 4)] == (102,) and preds[(5, 3, 1)][1] == 101 and preds[(5, 3, 1)][0] in (100, 103)


@pytest.mark.gpu
def test_bfs_depth_limit():
    edges = _edges()
    walks = _walks(_bfs_walk_steps(edges, cudf.Series([5], dtype="int32"), walk_depth=1, random_state=0))
    assert sorted(v for v, _ in walks.values()) == [(5, 3), (5, 4)]


@pytest.mark.gpu
def test_bfs_parallel_edges_one_predicate_drawn_uniformly():
    edges = _edges()
    n = 400
    steps = _bfs_walk_steps(edges, cudf.Series([5] * n, dtype="int32"), walk_depth=3, random_state=0)
    walks = _walks(steps)
    assert len(walks) == 2 * n and len(steps) == 3 * n
    first = np.array([p[0] for v, p in walks.values() if v == (5, 3, 1)])
    assert set(first.tolist()) == {100, 103}
    assert abs((first == 100).mean() - 0.5) < 5 * math.sqrt(0.25 / n)


@pytest.mark.gpu
def test_bfs_skipgram_pairs_not_duplicated():
    """Walks [5, p, 3, 101, 1] and [5, 102, 4] with window 2 give 14 + 6 pairs."""
    edges = _edges()
    pairs = SingleGPUWalkCorpus(_graph(edges), window_size=2).bfs_walk(
        edge_df=edges, walk_vertices=cudf.Series([5], dtype="int32"), walk_depth=3,
        random_state=0, word2vec_model="skipgram", min_count=1,
    )
    assert len(pairs) == 20


@pytest.mark.gpu
def test_bfs_roots_without_out_edges():
    edges = _edges()
    walks = _walks(_bfs_walk_steps(edges, cudf.Series([3, 1], dtype="int32"), walk_depth=3, random_state=0))
    # 3 -> 1 -> 5 -> {4}: the back edge leads on, and the tree stops at 4 (self-loop ignored)
    assert sorted(v for v, _ in walks.values()) == [(1, 5, 3), (1, 5, 4), (3, 1, 5, 4)]
    sink_edges = cudf.DataFrame({"subject": cudf.Series([0], dtype="int32"),
                                 "predicate": cudf.Series([9], dtype="int32"),
                                 "object": cudf.Series([7], dtype="int32")})
    with pytest.raises(ValueError, match="no walks"):
        _bfs_walk_steps(sink_edges, cudf.Series([7], dtype="int32"), walk_depth=2)
