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


def _rows(vertices, predicates):
    """(vertex path, predicate path) per row of BFS walk matrices, cut at -1."""
    out = []
    for v, p in zip(vertices.get().tolist(), predicates.get().tolist()):
        end = v.index(-1) if -1 in v else len(v)
        out.append((tuple(v[:end]), tuple(p[: end - 1])))
    return out


def _bfs_matrices(edges, roots, depth, random_state=0, walks_per_chunk=None):
    import cupy as cp

    chunks = list(SingleGPUWalkCorpus(_graph(edges), window_size=2).bfs_chunks(
        edges, cudf.Series(roots, dtype="int32"), walk_depth=depth, random_state=random_state,
        walks_per_chunk=walks_per_chunk,
    ))
    for v, p in chunks:
        assert v.dtype == cp.int32 and p.dtype == cp.int32
        assert v.shape[1] == depth + 1 and p.shape[1] == depth
    return chunks


@pytest.mark.gpu
def test_bfs_chunks_hold_the_root_to_leaf_paths_of_the_step_rows():
    """The walk matrices hold the same root-to-leaf paths as `_bfs_walk_steps`,
    padded with -1 vertices (and 0 predicates) after a path's end, in chunks
    of whole roots of at least `walks_per_chunk` walks."""
    import cupy as cp

    edges = _edges()
    chunks = _bfs_matrices(edges, [5, 3, 1], depth=3, walks_per_chunk=2)
    assert [len(v) for v, _ in chunks] == [2, 3]  # root 5; roots 3 and 1
    vertices = cp.concatenate([v for v, _ in chunks])
    predicates = cp.concatenate([p for _, p in chunks])
    rows = _rows(vertices, predicates)
    reference = _walks(_bfs_walk_steps(edges, cudf.Series([5, 3, 1], dtype="int32"), walk_depth=3, random_state=0))
    assert sorted(v for v, _ in rows) == sorted(v for v, _ in reference.values())
    legit = {(int(s), int(o), int(p)) for s, p, o in edges.to_pandas().itertuples(index=False)}
    for path, preds in rows:
        assert all((path[k], path[k + 1], preds[k]) in legit for k in range(len(preds)))
    ended = vertices[:, 1:] == -1
    assert bool((predicates[ended] == 0).all())


@pytest.mark.gpu
def test_bfs_chunks_draw_parallel_edge_predicates_uniformly_and_reproducibly():
    edges = _edges()
    n = 400
    first = [p[0] for v, p in _rows(*_bfs_matrices(edges, [5] * n, depth=3)[0]) if v == (5, 3, 1)]
    assert len(first) == n and set(first) == {100, 103}
    assert abs(np.mean(np.array(first) == 100) - 0.5) < 5 * math.sqrt(0.25 / n)
    again = _rows(*_bfs_matrices(edges, [5] * n, depth=3)[0])
    other = _rows(*_bfs_matrices(edges, [5] * n, depth=3, random_state=1)[0])
    assert again == _rows(*_bfs_matrices(edges, [5] * n, depth=3)[0]) and again != other


@pytest.mark.gpu
def test_bfs_chunks_without_any_walk_raise():
    sink_edges = cudf.DataFrame({"subject": cudf.Series([0], dtype="int32"),
                                 "predicate": cudf.Series([9], dtype="int32"),
                                 "object": cudf.Series([7], dtype="int32")})
    with pytest.raises(ValueError, match="no walks"):
        _bfs_matrices(sink_edges, [7], depth=2)
